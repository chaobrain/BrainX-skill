// Retry, timeout, rate-limit, and response guards adapt cyanheads MCP services
// (Apache-2.0); modified for this Node 18 resolver. See ../REUSE.md.

import { assertPublicHttpUrl } from './identifiers.mjs';

export class HttpError extends Error {
  constructor(message, options = {}) {
    super(message, options.cause ? { cause: options.cause } : undefined);
    this.name = 'HttpError';
    this.status = options.status;
    this.retryable = Boolean(options.retryable);
    this.retryAfter = options.retryAfter;
    this.url = options.url;
  }
}

function parseRetryAfter(value) {
  if (!value) return undefined;
  if (/^\d+$/.test(value.trim())) return Number(value.trim());
  const date = Date.parse(value);
  if (Number.isNaN(date)) return undefined;
  return Math.max(0, Math.round((date - Date.now()) / 1000));
}

function abortAfter(timeoutMs, externalSignal) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(new Error(`Request timed out after ${timeoutMs} ms`)), timeoutMs);
  const abort = () => controller.abort(externalSignal?.reason);
  externalSignal?.addEventListener('abort', abort, { once: true });
  return {
    signal: controller.signal,
    dispose() {
      clearTimeout(timeout);
      externalSignal?.removeEventListener('abort', abort);
    },
  };
}

async function sleep(milliseconds, signal) {
  if (milliseconds <= 0) return;
  await new Promise((resolve, reject) => {
    const timeout = setTimeout(resolve, milliseconds);
    const abort = () => {
      clearTimeout(timeout);
      reject(signal.reason ?? new Error('Request aborted'));
    };
    signal?.addEventListener('abort', abort, { once: true });
  });
}

export class HttpClient {
  constructor(options = {}) {
    this.fetch = options.fetchImpl ?? globalThis.fetch;
    this.timeoutMs = options.timeoutMs ?? Number(process.env.FULLTEXT_HTTP_TIMEOUT_MS || 30_000);
    this.maxBytes = options.maxBytes ?? Number(process.env.FULLTEXT_MAX_BYTES || 8_000_000);
    this.retries = options.retries ?? 2;
    this.userAgent = options.userAgent
      ?? process.env.FULLTEXT_RESOLVER_USER_AGENT
      ?? 'brainx-fulltext-resolver/1.0 (+https://github.com/chaobrain/BrainX-skill)';
  }

  async request(urlValue, options = {}) {
    let lastError;
    for (let attempt = 0; attempt <= this.retries; attempt += 1) {
      try {
        const response = await this.#fetchFollowingRedirects(urlValue, options);
        if (response.ok) return response;

        const retryAfter = parseRetryAfter(response.headers.get('retry-after'));
        const retryable = response.status === 429 || response.status >= 500;
        const error = new HttpError(`HTTP ${response.status} from ${response.url || urlValue}`, {
          status: response.status,
          retryable,
          retryAfter,
          url: response.url || String(urlValue),
        });
        if (!retryable || attempt === this.retries) throw error;
        lastError = error;
        const waitMs = retryAfter === undefined
          ? 250 * (2 ** attempt)
          : Math.min(retryAfter * 1000, 10_000);
        await sleep(waitMs, options.signal);
      } catch (error) {
        if (error instanceof HttpError && !error.retryable) throw error;
        if (options.signal?.aborted) throw options.signal.reason ?? error;
        lastError = error;
        if (attempt === this.retries) {
          if (error instanceof HttpError) throw error;
          throw new HttpError(`Request failed for ${urlValue}: ${error.message}`, {
            cause: error,
            retryable: true,
            url: String(urlValue),
          });
        }
        await sleep(250 * (2 ** attempt), options.signal);
      }
    }
    throw lastError;
  }

  async #fetchFollowingRedirects(urlValue, options) {
    let url = assertPublicHttpUrl(urlValue);
    for (let redirectCount = 0; redirectCount <= 5; redirectCount += 1) {
      const timed = abortAfter(this.timeoutMs, options.signal);
      try {
        const response = await this.fetch(url, {
          headers: {
            accept: options.accept ?? '*/*',
            'user-agent': this.userAgent,
            ...options.headers,
          },
          redirect: 'manual',
          signal: timed.signal,
        });
        if (![301, 302, 303, 307, 308].includes(response.status)) return response;
        const location = response.headers.get('location');
        if (!location) return response;
        url = assertPublicHttpUrl(new URL(location, url).toString());
      } finally {
        timed.dispose();
      }
    }
    throw new HttpError(`Too many redirects from ${urlValue}`, {
      retryable: false,
      url: String(urlValue),
    });
  }

  async text(url, options = {}) {
    const response = await this.request(url, options);
    const contentLength = Number(response.headers.get('content-length') || 0);
    if (contentLength > this.maxBytes) {
      throw new HttpError(`Response exceeds ${this.maxBytes} bytes: ${response.url || url}`, {
        retryable: false,
        url: response.url || String(url),
      });
    }
    const buffer = Buffer.from(await response.arrayBuffer());
    if (buffer.byteLength > this.maxBytes) {
      throw new HttpError(`Response exceeds ${this.maxBytes} bytes: ${response.url || url}`, {
        retryable: false,
        url: response.url || String(url),
      });
    }
    return {
      body: buffer.toString('utf8'),
      contentType: response.headers.get('content-type') || '',
      url: response.url || String(url),
    };
  }

  async json(url, options = {}) {
    const response = await this.text(url, {
      ...options,
      accept: options.accept ?? 'application/json',
    });
    if (/^\s*<(!doctype\s+html|html[\s>])/i.test(response.body)) {
      throw new HttpError(`Expected JSON but received HTML from ${response.url}`, {
        retryable: true,
        url: response.url,
      });
    }
    try {
      return JSON.parse(response.body);
    } catch (error) {
      throw new HttpError(`Invalid JSON from ${response.url}`, {
        cause: error,
        retryable: true,
        url: response.url,
      });
    }
  }
}
