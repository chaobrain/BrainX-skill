import { createInterface } from 'node:readline';

const threads = new Map();
const turns = new Map();
const approvals = new Map();
let nextThread = 1;
let nextTurn = 1;
const write = (message) => process.stdout.write(`${JSON.stringify(message)}\n`);
function finish(threadId, turnId, prompt, decision) {
  if (prompt === 'empty') {
    write({ method: 'turn/completed', params: { threadId, turn: { id: turnId, status: 'completed' } } });
    return;
  }
  const thread = threads.get(threadId);
  write({ method: 'item/completed', params: { threadId, turnId,
    item: { type: 'agentMessage', id: 'comment', phase: 'commentary', text: 'Ignore commentary' } } });
  // An event from another turn must never enter the report.
  write({ method: 'item/completed', params: { threadId, turnId: 'unrelated',
    item: { type: 'agentMessage', id: 'wrong', text: 'Wrong turn' } } });
  write({ method: 'item/completed', params: { threadId, turnId,
    item: { type: 'agentMessage', id: 'report', phase: 'final_answer',
      text: JSON.stringify({ thread, turn: turns.get(turnId), decision,
        argv: process.argv.slice(2) }) } } });
  write({ method: 'turn/completed', params: { threadId, turn: { id: turnId, status: 'completed' } } });
}

const input = createInterface({ input: process.stdin, crlfDelay: Infinity });
input.on('line', (line) => {
  const { id, method, params, result } = JSON.parse(line);
  if (approvals.has(id)) {
    const turn = approvals.get(id); finish(...turn, result.decision); approvals.delete(id); return;
  }
  if (method === 'initialize') { write({ id, result: {} }); return; }
  if (method === 'config/read') {
    const servers = process.env.TEST_NO_MCP ? {} : { codex: { command: 'node' } };
    if (params.cwd) servers.project_mcp = { command: 'node' };
    write({ id, result: { config: { mcp_servers: servers } } }); return;
  }
  if (method === 'thread/start') {
    const threadId = `thread-${nextThread++}`;
    threads.set(threadId, { start: params, history: [] });
    write({ id, result: { thread: { id: threadId } } }); return;
  }
  if (method === 'thread/resume') {
    const thread = threads.get(params.threadId);
    if (!thread) { write({ id, error: { code: -32000, message: 'Thread not found' } }); return; }
    thread.resume = params;
    write({ id, result: { thread: { id: params.threadId } } }); return;
  }
  if (method === 'thread/read') {
    const thread = threads.get(params.threadId);
    if (!thread) { write({ id, error: { code: -32000, message: 'Thread not found' } }); return; }
    write({ id, result: { thread: { id: params.threadId, cwd: thread.start.cwd } } }); return;
  }
  if (method === 'turn/start') {
    const { threadId } = params;
    const turnId = `turn-${nextTurn++}`;
    const prompt = params.input[0].text;
    threads.get(threadId).history.push(prompt);
    turns.set(turnId, params);
    if (prompt === 'crash') { process.exit(1); }
    if (prompt !== 'early') write({ id, result: { turn: { id: turnId } } });
    if (prompt === 'stall') return;
    if (prompt === 'approval') {
      const approvalId = `approval-${turnId}`;
      approvals.set(approvalId, [threadId, turnId, prompt]);
      write({ id: approvalId, method: 'item/commandExecution/requestApproval', params: { threadId, turnId } });
      return;
    }
    if (prompt === 'fail') {
      write({ method: 'turn/completed', params: { threadId,
        turn: { id: turnId, status: 'failed', error: { message: 'Model unavailable' } } } }); return;
    }
    finish(threadId, turnId, prompt);
    if (prompt === 'early') write({ id, result: { turn: { id: turnId } } });
    return;
  }
  if (method === 'turn/interrupt') {
    write({ id, result: {} });
    write({ method: 'turn/completed', params: { threadId: params.threadId,
      turn: { id: params.turnId, status: 'interrupted' } } });
  }
});
