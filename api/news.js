// api/news.js
const axios = require("axios");
require("dotenv").config();

module.exports = async (req, res) => {
  try {
    const response = await axios.get("https://newsapi.org/v2/top-headlines", {
      params: {
        category: "business",
        country: "us",
        apiKey: process.env.NEWS_API_KEY,
      },
    });

    const topArticles = response.data.articles.slice(0, 5);
    res.status(200).json({ news: topArticles });
  } catch (err) {
    res.status(500).json({ error: "News API failed", detail: err.message });
  }
};
