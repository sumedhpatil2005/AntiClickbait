console.log("Extension content script loaded!");
function createSummarizeButton(url) {
  const element = document.createElement('div');
  element.textContent = 'Summarize';
  element.className = 'summary-button';
  element.onclick = async function (e) {
    e.preventDefault();
    e.stopImmediatePropagation();
    try {
      element.textContent = 'Loading...';
      const res = await fetch('http://localhost:8080/summary', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url }),
      });
      if (res.status !== 200) {
        throw new Error('Could not summarize');
      }
      const { summary } = await res.json();
      element.textContent = summary;
    } catch (err) {
      element.textContent = 'Error summarizing';
    }
  };
  return element;
}

function findAndEnrichLinks() {
  const currentHost = window.location.host;
  const links = document.querySelectorAll('a');
  for (const link of [...links]) {
    // ignore already annotated buttons
    if (link.getAttribute('data-summary') === '1') {
      continue;
    }
    const url = new URL(link.getAttribute('href'), window.location.origin);
    if (url.origin !== 'null' && url.host !== currentHost) {
      //link.parentNode.append(createSummarizeButton(link.getAttribute('href')));
      // mark as annotated
      link.setAttribute('data-summary', '1');
    }
  }
}

// run on page load
findAndEnrichLinks();

// Create an observer to run on page changes
const observer = new MutationObserver(() => findAndEnrichLinks());

// Start observing the target node for configured mutations
observer.observe(document, {
  attributes: true,
  childList: true,
  subtree: true,
});
setTimeout(() => {
  console.log("Checking clickbait...");
  checkClickbait();
}, 4000);
async function checkClickbait() {
  try {
    console.log("Checking clickbait...");

    const ytTitleNode = document.querySelector("h1.ytd-watch-metadata");

    if (!ytTitleNode) {
      console.log("Title not found yet");
      return;
    }

    // Prevent duplicate badges
    if (document.querySelector(".clickbait-badge")) return;

    const videoTitle = ytTitleNode.innerText;

    const res = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        title: videoTitle,
        description: "",
        thumbnail_text: ""
      })
    });

    const data = await res.json();
    console.log("Clickbait API Result:", data);

    const badge = document.createElement("div");
    badge.className = "clickbait-badge";

    // 🎨 Dynamic color
    const bgColor = data.prediction === "Misleading" ? "#e74c3c" : "#2ecc71";

    badge.style.background = bgColor;
    badge.style.color = "#fff";
    badge.style.padding = "10px";
    badge.style.marginTop = "8px";
    badge.style.borderRadius = "8px";
    badge.style.fontSize = "14px";
    badge.style.fontWeight = "bold";
    badge.style.fontFamily = "Arial, sans-serif";
    badge.style.boxShadow = "0 2px 6px rgba(0,0,0,0.2)";
    badge.style.transition = "0.3s ease";

    badge.textContent =
      `Clickbait Check: ${data.prediction} (Confidence: ${(data.confidence * 100).toFixed(1)}%)`;

    // 📊 Confidence bar
    const barContainer = document.createElement("div");
    barContainer.style.background = "rgba(255,255,255,0.3)";
    barContainer.style.height = "6px";
    barContainer.style.borderRadius = "3px";
    barContainer.style.marginTop = "6px";

    const bar = document.createElement("div");
    bar.style.height = "6px";
    bar.style.width = `${data.confidence * 100}%`;
    bar.style.background = "#fff"; 
    bar.style.borderRadius = "3px";
    bar.style.transition = "width 0.5s ease";

    barContainer.appendChild(bar);
    badge.appendChild(barContainer);

    // 💡 Tooltip
    badge.title =
      "Prediction based on title patterns, emotional triggers, and clickbait signals.";

    ytTitleNode.parentNode.appendChild(badge);

  } catch (err) {
    console.error("API Error:", err);
  }
}


setTimeout(checkClickbait, 4000);
