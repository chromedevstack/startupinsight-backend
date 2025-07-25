// api/rate-idea.js
const axios = require("axios");
require("dotenv").config();

module.exports = async (req, res) => {
  const { idea, businessType, targetMarket } = req.query;

  if (!idea) {
    return res.status(400).json({ error: "Missing startup idea input." });
  }

  const prompt = `
Rate this startup idea:
Idea: ${idea}
Business Type: ${businessType || "Not Specified"}
Target Market: ${targetMarket || "Not Specified"}

Give a short summary, risk analysis, potential profitability (Low, Medium, High), and a 1-10 score.
`;

  try {
    const response = await axios.post(
      "https://api.openai.com/v1/chat/completions",
      {
        model: "gpt-3.5-turbo",
        messages: [{ role: "user", content: prompt }],
      },
      {
        headers: {
          Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
        },
      }
    );

    const aiOutput = response.data.choices[0].message.content;
    res.status(200).json({ result: aiOutput });
  } catch (err) {
    res.status(500).json({ error: "AI response failed", detail: err.message });
  }
};
