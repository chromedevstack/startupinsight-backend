const API_URL = "const API_URL = "https://startupinsight-backend.x11cr24.repl.co/generate";

document.getElementById("analyzeBtn").addEventListener("click", async () => {
  const prompt = document.getElementById("prompt").value.trim();
  if (!prompt) {
    alert("Please enter a prompt!");
    return;
  }
  const outputElem = document.getElementById("output");
  outputElem.textContent = "Loading...";

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt }),
    });
    if (!response.ok) throw new Error("Network response was not ok");

    const data = await response.json();
    outputElem.textContent = data.output;
  } catch (error) {
    outputElem.textContent = "Error: " + error.message;
  }
});
