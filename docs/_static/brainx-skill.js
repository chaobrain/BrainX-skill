document.addEventListener("DOMContentLoaded", () => {
  const video = document.querySelector(".quickstart-media video");
  if (!video) return;

  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (reduceMotion) {
    video.autoplay = false;
    video.pause();
  }
});
