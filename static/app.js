const form = document.getElementById("searchForm");
const statusEl = document.getElementById("status");
const results = document.getElementById("results");
const summary = document.getElementById("summary");
const grid = document.getElementById("grid");

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  statusEl.textContent = "Analyzing video... Please wait.";
  results.classList.add("hidden");
  grid.innerHTML = "";
  const data = new FormData(form);

  try {
    const response = await fetch("/api/search", { method: "POST", body: data });
    const json = await response.json();
    if (!response.ok) throw new Error(json.error || "Search failed");

    summary.textContent = `${json.candidates.length} possible sighting(s) found at ${json.camera} — ${json.location}.`;

    if (json.candidates.length === 0) {
      grid.innerHTML = "<p>No candidate was found above the selected threshold.</p>";
    } else {
      json.candidates.forEach(c => {
        const div = document.createElement("div");
        div.className = "item";
        div.innerHTML = `
          <img src="${c.image}" alt="Possible sighting">
          <p><b>Camera:</b> ${c.camera}</p>
          <p><b>Location:</b> ${c.location}</p>
          <p><b>Video time:</b> ${c.time_seconds}s</p>
          <p><b>Similarity:</b> ${c.score}</p>
        `;
        grid.appendChild(div);
      });
    }
    results.classList.remove("hidden");
    statusEl.textContent = "Analysis completed.";
  } catch (err) {
    statusEl.textContent = "Error: " + err.message;
  }
});
