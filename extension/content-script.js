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
    badge.style.background = "#222";
    badge.style.color = "#fff";
    badge.style.padding = "8px";
    badge.style.marginTop = "8px";
    badge.style.borderRadius = "6px";
    badge.style.fontSize = "14px";

    badge.textContent = `Clickbait Check: ${data.prediction} (Confidence: ${data.confidence.toFixed(2)})`;

    ytTitleNode.parentNode.appendChild(badge);

  } catch (err) {
    console.error("API Error:", err);
  }
}

setTimeout(checkClickbait, 4000);
