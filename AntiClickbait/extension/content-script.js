console.log("Anti-Clickbait Extension loaded!");

// Flag to prevent duplicate badges
let badgeAdded = false;
let lastUrl = location.href;

/**
 * Extract video ID from YouTube URL
 */
function getVideoId() {
  const url = new URL(window.location.href);

  // Standard watch URL: youtube.com/watch?v=VIDEO_ID
  if (url.pathname === '/watch') {
    return url.searchParams.get('v');
  }

  // Shorts URL: youtube.com/shorts/VIDEO_ID
  if (url.pathname.startsWith('/shorts/')) {
    return url.pathname.split('/shorts/')[1];
  }

  return null;
}

/**
 * Get YouTube title from the page (fallback if API fails)
 */
function getYouTubeTitle() {
  const selectors = [
    'h1.ytd-watch-metadata yt-formatted-string',
    'h1.ytd-watch-metadata',
    '#title h1 yt-formatted-string',
    '#title h1',
    'ytd-watch-metadata h1'
  ];

  for (const selector of selectors) {
    const element = document.querySelector(selector);
    if (element && element.innerText && element.innerText.trim()) {
      return { element, text: element.innerText.trim() };
    }
  }
  return null;
}

/**
 * Remove existing badges (called on navigation)
 */
function removeBadges() {
  const badges = document.querySelectorAll('.clickbait-badge');
  badges.forEach(b => b.remove());
  badgeAdded = false;
  console.log("Removed old badges");
}

/**
 * Main function to check if video is clickbait
 */
async function checkClickbait() {
  // 1. Check if badge already exists (to avoid duplicates during retries)
  if (document.querySelector('.clickbait-badge')) {
    console.log("Badge already exists, skipping...");
    badgeAdded = true;
    return true;
  }

  // 2. Get Video ID
  const videoId = getVideoId();
  if (!videoId) {
    console.log("Not on a video page");
    return false;
  }

  // 3. Find Title Element (Placement Target)
  const titleResult = getYouTubeTitle();
  if (!titleResult) {
    console.log("Title element not found yet...");
    return false;
  }

  try {
    console.log(`Checking clickbait for ${videoId}...`);

    // 4. Call API
    const res = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ video_id: videoId })
    });

    const data = await res.json();
    if (data.error) throw new Error(data.error);

    // 5. Create Badge
    const badge = document.createElement("div");
    badge.className = "clickbait-badge";

    // Styling based on prediction
    const isMisleading = data.prediction === "Misleading";
    badge.style.background = isMisleading ? "#dc2626" : "#16a34a"; // Red or Green
    badge.style.color = "#fff";
    badge.style.padding = "8px 12px";
    badge.style.marginTop = "10px";
    badge.style.borderRadius = "6px";
    badge.style.fontSize = "14px";
    badge.style.fontWeight = "600";
    badge.style.display = "inline-flex";
    badge.style.alignItems = "center";
    badge.style.gap = "8px";
    badge.style.boxShadow = "0 2px 4px rgba(0,0,0,0.2)";
    badge.style.cursor = "default";

    // Text Content
    const icon = isMisleading ? "⚠️" : "✅";
    const conf = (data.confidence * 100).toFixed(1);
    badge.textContent = `${icon} ${data.prediction} (${conf}%)`;

    if (data.reason) {
      badge.title = `Reason: ${data.reason}`;
    }

    // 6. Append Badge
    const { element: ytTitleNode } = titleResult;
    // Try to append after the h1 or inside the metadata container
    const parent = ytTitleNode.closest('h1') || ytTitleNode.parentNode;
    parent.appendChild(badge);

    badgeAdded = true;
    console.log("Badge added!");
    return true;

  } catch (err) {
    console.error("API Error:", err);
    return false;
  }
}

/**
 * Init with retry logic
 */
async function initClickbaitCheck() {
  // Clear any existing badges first
  removeBadges();

  const delays = [500, 1500, 3000, 5000];
  for (const delay of delays) {
    await new Promise(r => setTimeout(r, delay));
    const success = await checkClickbait();
    if (success) return;
  }
}

/**
 * Handle YouTube SPA Navigation
 */
setInterval(() => {
  if (location.href !== lastUrl) {
    console.log(`URL changed: ${lastUrl} -> ${location.href}`);
    lastUrl = location.href;
    initClickbaitCheck();
  }
}, 1000); // Check every second for URL change (more robust than MutationObserver for history changes)

// Initial Run
initClickbaitCheck();
