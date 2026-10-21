// Negative-Space Reader: the demo's motion (plan 7.1). A lens cursor with a particle trail (pointer devices only, no
// trail under reduced motion), click-to-select on ghost paragraphs (writes the type into a hidden Gradio textbox,
// which the Python handler reads), and count-up numbers when a panel is redrawn. Nothing here touches the document
// text or sends anything anywhere.
(() => {
  const fine = matchMedia("(pointer: fine)").matches;
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const colors = ["#FF8C52", "#359FA0", "#8AD6D1", "#1d7374"];

  // background layers
  const aurora = document.createElement("div");
  aurora.className = "nsr-aurora";
  aurora.innerHTML = "<i></i><i></i><i></i>";
  const grain = document.createElement("div");
  grain.className = "nsr-grain";
  document.body.prepend(grain);
  document.body.prepend(aurora);

  // ghost click -> Evidence panel (the handler is wired in Python on the hidden textbox #nsr-pick)
  document.addEventListener("click", (e) => {
    const g = e.target.closest(".nsr-ghost");
    if (!g) return;
    const ta = document.querySelector("#nsr-pick textarea, #nsr-pick input");
    if (!ta) return;
    document.querySelectorAll(".nsr-ghost.is-selected").forEach((x) => x.classList.remove("is-selected"));
    g.classList.add("is-selected");
    ta.value = g.dataset.type + "|" + Date.now();
    ta.dispatchEvent(new Event("input", { bubbles: true }));  // updates the box's bound value
    document.querySelector("#nsr-pick-go")?.click();            // a button press is what reaches the backend
  });

  // count-up for numbers marked data-count (percentages in the ghosts and panels)
  const counted = new WeakSet();
  const countUp = (el) => {
    if (counted.has(el) || reduce) return;
    counted.add(el);
    const end = parseFloat(el.dataset.count), suffix = el.dataset.suffix || "", t0 = performance.now(), dur = 900;
    const step = (t) => {
      const k = Math.min(1, (t - t0) / dur), v = end * (1 - Math.pow(1 - k, 3));
      el.textContent = (Number.isInteger(end) ? Math.round(v) : v.toFixed(2)) + suffix;
      if (k < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  };
  new MutationObserver(() => document.querySelectorAll("[data-count]").forEach(countUp)).observe(document.body, { childList: true, subtree: true });

  if (!fine) return;
  document.documentElement.classList.add("nsr-cursor");
  const ring = document.createElement("div");
  ring.className = "nsr-ring-cursor";
  const dot = document.createElement("div");
  dot.className = "nsr-dot";
  document.body.append(ring, dot);
  let mx = innerWidth / 2, my = innerHeight / 2, rx = mx, ry = my;
  const hoverable = "button, a, [role=button], .nsr-ghost, input[type=range], select, summary, .wrap-inner, .nsr-quote, .tab-nav button, label";
  addEventListener("pointermove", (e) => {
    mx = e.clientX; my = e.clientY;
    dot.style.transform = `translate(${mx}px, ${my}px)`;
    ring.classList.toggle("is-hover", !!e.target.closest(hoverable));
    spawn(e);
  }, { passive: true });
  addEventListener("pointerdown", () => ring.classList.add("is-down"));
  addEventListener("pointerup", () => ring.classList.remove("is-down"));
  document.addEventListener("mouseleave", () => { ring.style.opacity = "0"; dot.style.opacity = "0"; });
  document.addEventListener("mouseenter", () => { ring.style.opacity = "1"; dot.style.opacity = "1"; });
  (function follow() {  // the ring closes most of the gap every frame: a hint of weight, no visible lag
    rx += (mx - rx) * 0.6; ry += (my - ry) * 0.6;
    ring.style.transform = `translate(${rx}px, ${ry}px)`;
    requestAnimationFrame(follow);
  })();

  // the trail: a canvas of additive particles behind the cursor
  let spawn = () => {};
  if (reduce) return;
  const canvas = document.createElement("canvas");
  canvas.className = "nsr-trail";
  document.body.append(canvas);
  const ctx = canvas.getContext("2d");
  const resize = () => {
    canvas.width = innerWidth * devicePixelRatio; canvas.height = innerHeight * devicePixelRatio;
    canvas.style.width = innerWidth + "px"; canvas.style.height = innerHeight + "px";
    ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  };
  resize();
  addEventListener("resize", resize);
  const P = [];
  let last = 0, lx = 0, ly = 0;
  spawn = (e) => {
    const now = performance.now(), speed = Math.hypot(e.clientX - lx, e.clientY - ly);
    lx = e.clientX; ly = e.clientY;
    if (now - last < 12) return;
    last = now;
    const n = 1 + Math.min(3, speed / 12 | 0);
    for (let i = 0; i < n; i++) {
      P.push({ x: e.clientX + (Math.random() - 0.5) * 6, y: e.clientY + (Math.random() - 0.5) * 6,
               vx: (Math.random() - 0.5) * 1.4, vy: (Math.random() - 0.5) * 1.4 - 0.35, life: 1,
               r: 1.5 + Math.random() * 2.5, c: colors[(Math.random() * colors.length) | 0] });
    }
    if (P.length > 260) P.splice(0, P.length - 260);
  };
  (function draw() {
    ctx.clearRect(0, 0, innerWidth, innerHeight);
    ctx.globalCompositeOperation = "source-over";  // additive blending washes out on a light page
    for (let i = P.length - 1; i >= 0; i--) {
      const p = P[i];
      p.x += p.vx; p.y += p.vy; p.vy += 0.01; p.life -= 0.028;
      if (p.life <= 0) { P.splice(i, 1); continue; }
      ctx.globalAlpha = Math.max(0, p.life) * 0.7;
      ctx.fillStyle = p.c; ctx.shadowBlur = 14; ctx.shadowColor = p.c;
      ctx.beginPath(); ctx.arc(p.x, p.y, p.r * p.life, 0, 6.2832); ctx.fill();
    }
    requestAnimationFrame(draw);
  })();
})();
