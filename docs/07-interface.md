# 7. Interface (`web/index.html`)

[← 6. Service and server](06-server.md) · [Guide index](../README.md#step-by-step-guide) · [8. Run on your computer →](08-run-locally.md)

![LayaMed interface](img/interface.jpg)

A single file with plain HTML, CSS and JavaScript: no library, no CDN. The page works offline and sends nothing outside the local server.

## Layout

```
┌──────────────── header (logo) ────────────────┐
│ Case field (textarea)      │ Evidence bars          │
│ hint + sections read       │ Result card            │  ← right column stays in view (sticky)
│ Example cases              │ Warnings               │
│ Text Laya receives         │                        │
└───────────────────────────────────────────────┘
```

- A two-column CSS grid (`5fr 4fr`); below 760 px wide it becomes a single column (phones).
- Colors as CSS tokens on `:root` (`--green #2E9461`, `--blue #1F6E9C`...), redefined for dark mode under `@media (prefers-color-scheme: dark)`. Text uses darker greens than fills (`--green-ink`) so small text meets the WCAG AA contrast; the bar colors come from `--bar-0` to `--bar-3`.
- The logo is embedded as SVG (and as a `data:` favicon), so it shows even when the file is opened from disk.

## The JavaScript, step by step

**1. Configuration.** On load, the page requests `GET /api/config` (classes, labels, example cases) and builds the 4 bar rows and the buttons. If the page was opened from disk (no server), it explains how to start the server and redirects itself as soon as the server answers.

**2. Analysis while typing.** Every `input` event restarts a 120 ms timer; when typing pauses, `analyze()` sends `{"text": field.value}` to `POST /api/analyze`.

```js
function schedule() { clearTimeout(timer); timer = setTimeout(analyzeContinuously, 120); }
```

Each request gets a number (`const mine = ++request`). A response is only drawn if it is still the latest one: late responses to older requests are discarded, so the screen never "goes back" to an earlier text.

**3. Drawing by state.** `drawCard(r)` has one branch per `r.state`:

| State | Card | Bars |
|---|---|---|
| `empty` | "Start writing..." | empty |
| `waiting` | current tokens / 100, progress bar, how many words are missing | empty |
| `out_of_scope` | "Not a clinical case / No diagnosis" + reason | empty |
| `result` | leading disease + probability, understanding, management, tokens, time | evidence per disease |

The bars are created once; afterwards only `width`, color and text change, and the CSS transition (`transition: width .35s`) makes them "grow". The color follows a gradient by value (light blue → blue → green).

**4. Text safety.** Only numbers and fixed labels go in as HTML. Everything that comes from the user (the cleaned text, the warning terms) goes in through `textContent`, never `innerHTML`.

**5. Single field with sections.** The user types `Physical examination:` and `Labs:` at the start of lines; the server splits the sections and returns `sections_read`, shown below the field ("Sections read: history · physical examination · labs").

**6. Example cases typed word by word.** Each button clears the field and types the case one word every 90 ms (0.5 s pause between sections), with Laya following along:

```js
for (const word of ex.text.match(/\S+\s*/g) || []) {
  await wait(/\n\n/.test(field.value.slice(-2)) ? MS_BETWEEN_SECTIONS : MS_PER_WORD);
  if (mine !== typingRun) return;          // the user typed, clicked Clear or another case
  field.value += word;
  analyzeContinuously();
}
```

`analyzeContinuously` (used both when typing by hand and by the automatic typing) keeps at most one request in flight: if new text arrives during a request, it analyzes only the latest version again when the request ends. This way the GPU never builds up a queue.

## Customizing

- **Texts and language:** all labels are in the HTML and in `drawCard`.
- **Example cases:** `EXAMPLES` and `EXAMPLE_LABELS` in `layamed/service.py` (history, exam, labs).
- **Screenshots and demo GIF:** with the server running on the real model, `python tools/capture_demo.py` writes `docs/img/interface.png`, `interface-dark.png` and `demo.gif` (see the script's docstring).
- **Colors and dark mode:** the tokens at the top of the `<style>` block (both themes).
- **Typing speed:** `MS_PER_WORD` and `MS_BETWEEN_SECTIONS`.
- **Bar scale:** `FULL_CASE_TOKENS` and `FULL_SECTION_TOKENS` in `service.py`. To show the plain probability in the bars, replace `r.evidence[c]` with `r.probabilities[c]` in `drawBars`.

---

[← 6. Service and server](06-server.md) · [Guide index](../README.md#step-by-step-guide) · [8. Run on your computer →](08-run-locally.md)
