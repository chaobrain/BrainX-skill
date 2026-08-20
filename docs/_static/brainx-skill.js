document.addEventListener("DOMContentLoaded", () => {
  const video = document.querySelector(".quickstart-media video");
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (video && reduceMotion) {
    video.autoplay = false;
    video.pause();
  }

  document.querySelectorAll(".prompt-bubble").forEach((prompt) => {
    const promptText = prompt.innerText.trim();
    if (!promptText) return;

    const button = document.createElement("button");
    button.type = "button";
    button.className = "prompt-copy-button";
    button.title = "Copy prompt";
    button.setAttribute("aria-label", "Copy prompt to clipboard");
    button.innerHTML = '<i class="fa-regular fa-copy" aria-hidden="true"></i><span>Copy</span>';
    prompt.classList.add("has-copy-button");
    prompt.append(button);

    button.addEventListener("click", async () => {
      try {
        if (navigator.clipboard && window.isSecureContext) {
          await navigator.clipboard.writeText(promptText);
        } else {
          const textarea = document.createElement("textarea");
          textarea.value = promptText;
          textarea.setAttribute("readonly", "");
          textarea.style.position = "fixed";
          textarea.style.opacity = "0";
          document.body.append(textarea);
          textarea.select();
          const copied = document.execCommand("copy");
          textarea.remove();
          if (!copied) throw new Error("Clipboard command failed");
        }

        button.classList.add("is-copied");
        button.innerHTML = '<i class="fa-solid fa-check" aria-hidden="true"></i><span>Copied</span>';
        button.setAttribute("aria-label", "Prompt copied to clipboard");
        window.setTimeout(() => {
          button.classList.remove("is-copied");
          button.innerHTML = '<i class="fa-regular fa-copy" aria-hidden="true"></i><span>Copy</span>';
          button.setAttribute("aria-label", "Copy prompt to clipboard");
        }, 1800);
      } catch (_error) {
        button.innerHTML = '<i class="fa-solid fa-xmark" aria-hidden="true"></i><span>Retry</span>';
        button.setAttribute("aria-label", "Copy failed. Retry copying prompt");
      }
    });
  });
});
