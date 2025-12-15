// MathJax configuration to render inline ($...$) and block ($$...$$) math.
window.MathJax = {
  tex: {
    inlineMath: [["$", "$"], ["\\(", "\\)"]],
    displayMath: [["$$", "$$"], ["\\[", "\\]"]],
    processEscapes: true,
    tags: "ams",
  },
  options: {
    skipHtmlTags: ["script", "noscript", "style", "textarea", "pre", "code"],
  },
};
