console.log("Clickbait Clarifier Extension loaded!");

// Flag to prevent duplicate badges
let badgeAdded = false;
let lastUrl = location.href;
let inFlightVideoId = null; // Prevent simultaneous requests for same ID

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

  // 2b. Prevent simultaneous requests for the same video
  if (inFlightVideoId === videoId) {
    console.log(`Request already in flight for ${videoId}, skipping...`);
    return false;
  }
  inFlightVideoId = videoId;

  // 3. Find Title Element (Placement Target)
  const titleResult = getYouTubeTitle();
  if (!titleResult) {
    console.log("Title element not found yet...");
    return false;
  }

  try {
    const startUrl = location.href; // Capture URL at start
    console.log(`Checking clickbait for ${videoId}...`);

    // 4. PRE-CHECK: Add "Checking..." status immediately
    let badge = document.querySelector('.clickbait-badge');
    if (!badge) {
      badge = document.createElement("div");
      badge.className = "clickbait-badge";
      badge.style.background = "#4b5563"; // Gray while checking
      badge.style.color = "#fff";
      badge.style.padding = "6px 10px";
      badge.style.marginTop = "12px";
      badge.style.borderRadius = "4px";
      badge.style.fontSize = "14px";
      badge.style.fontFamily = "Roboto, Arial, sans-serif";
      badge.style.fontWeight = "500";
      badge.style.display = "inline-flex";
      badge.style.alignItems = "center";
      badge.style.gap = "6px";
      badge.style.width = "fit-content";
      badge.style.cursor = "wait";
      badge.style.zIndex = "999";
      badge.textContent = "⏱️ Checking for Clickbait...";
      
      const { element: ytTitleNode } = titleResult;
      const titleContainer = ytTitleNode.closest('ytd-watch-metadata') || ytTitleNode.parentNode;
      const h1Element = titleContainer.querySelector('h1') || ytTitleNode;
      h1Element.insertAdjacentElement('afterend', badge);
    }

    // 5. Call API
    const res = await fetch("https://anticlickbait.onrender.com/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ video_id: videoId }),
      signal: AbortSignal.timeout(15000) // 15s timeout
    });

    if (res.status === 503 || res.status === 502) {
      badge.textContent = "⏳ Server Waking Up (Wait 30s)...";
      console.log("Server is booting up...");
      throw new Error("Server Booting");
    }

    // RACE CONDITION FIX: If URL changed while fetching, abort.
    if (location.href !== startUrl) {
      console.log("URL changed during fetch, discarding old result.");
      inFlightVideoId = null; // Release lock
      return false;
    }

    const data = await res.json();
    if (data.error) throw new Error(data.error);

    // 6. Update Badge with Results
    const isMisleading = data.prediction === "Misleading";
    badge.style.background = isMisleading ? "#dc2626" : "#16a34a"; // Red or Green
    badge.style.cursor = "default";

    // Text Content
    const icon = "⚠️";
    const conf = (data.confidence * 100).toFixed(0);

    let text = isMisleading
      ? `${icon} Misleading (${conf}%)`
      : `Authenticity Score: ${conf}%`;

    // Add Timestamp if verified
    if (data.verification_timestamp && !isMisleading) {
      text += ` • 🎯 Key Moment at ${data.verification_timestamp}`;
    }

    badge.textContent = text;

    if (data.reason) {
      badge.title = `Reason: ${data.reason}`;
    }

    badge.setAttribute('data-video-id', videoId); // Mark badge with ID
    badgeAdded = true;
    console.log("Badge updated with result!");
    inFlightVideoId = null; // Release lock
    return true;

  } catch (err) {
    if (err.name === 'TimeoutError' || err.message === 'Server Booting') {
        const badge = document.querySelector('.clickbait-badge');
        if (badge) badge.textContent = "⌛ Still waking up... please wait.";
    }
    console.error("API Error:", err);
    inFlightVideoId = null; // Release lock
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
