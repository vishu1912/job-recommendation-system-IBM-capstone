// ===== Part 2: PDF Resume Job Recommendation System =====

// Part 2 Task 2: Import and Initialize
const fs = require("fs");
const pdf = require("pdf-parse");
const { HfInference } = require("@huggingface/inference");
const readline = require("readline");
const { ChromaClient } = require("chromadb");

// Create an instance of the Hugging Face inference client (name it `hf`)
const hf = new HfInference("YOUR_API_KEY"); // <-- put your real HF key

// Initialize your database (Chroma) and collection name
const chroma = new ChromaClient({ path: "http://localhost:8000" });
const collectionName = "job_collection"; // reuse the same as Part 1

// Require the array content of jobPostings.js
const jobPostings = require("./jobPostings.js");

// ---------- Part 2 Task 3: Extract Text and Generate Embeddings ----------

// A. extractTextFromPDF(): extract text + normalize (newline -> space, collapse spaces)
async function extractTextFromPDF(filePath) {
  const dataBuffer = fs.readFileSync(filePath);
  const parsed = await pdf(dataBuffer);
  const text = (function (t) {
    return (t || "").replace(/\n+/g, " ").replace(/\s{2,}/g, " ").trim();
  })(parsed.text);
  return text;
}

// B. generateEmbeddings(): use HF inference (MiniLM-L6-v2)
async function generateEmbeddings(text) {
  const res = await hf.featureExtraction({
    model: "sentence-transformers/all-MiniLM-L6-v2",
    inputs: text,
  });
  // Normalize to a single float[] vector
  const vec = Array.isArray(res[0]) ? res[0] : res;
  return vec.map(Number);
}

// ---------- Part 2 Task 4: Get the Resume ----------

// A. promptUserInput(): prompt for the PDF path (CLI)
function promptUserInput() {
  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
  return new Promise((resolve) => {
    rl.question("Enter the path to your resume PDF (e.g., ./testResume.pdf): ", (answer) => {
      rl.close();
      resolve(answer.trim());
    });
  });
}

// B. extractFromPDF(): separate helper that returns text from a PDF path (uses extractTextFromPDF)
async function extractFromPDF(pdfPath) {
  return await extractTextFromPDF(pdfPath);
}

// ---------- Part 2 Task 5: Store Embeddings ----------

// storeEmbeddings(): store job post embeddings into Chroma (per the sample steps)
async function storeEmbeddings(jobPostingsArr) {
  const jobEmbeddings = [];
  const metadatas = jobPostingsArr.map(() => ({})); // empty metadata objects

  // iterate and push embeddings (lowercased text)
  for (const job of jobPostingsArr) {
    const embedding = await generateEmbeddings(String(job.jobDescription || "").toLowerCase());
    jobEmbeddings.push(embedding);
  }

  // ids are index strings; documents are job titles
  const ids = jobPostingsArr.map((_, index) => index.toString());
  const jobTexts = jobPostingsArr.map((job) => job.jobTitle);

  try {
    const collection = await chroma.getOrCreateCollection({ name: collectionName });
    await collection.add({
      ids,
      documents: jobTexts,
      embeddings: jobEmbeddings,
      metadatas,
    });
    console.log("Stored embeddings in Chroma DB.");
  } catch (error) {
    console.error("Error storing embeddings in Chroma DB:", error);
    throw error;
  }
}

// ---------- Part 2 Task 6: Create the Main Function ----------

async function main() {
  try {
    // Store the job post embeddings first
    await storeEmbeddings(jobPostings);

    // Prompt user for PDF path (test with the provided testResume.pdf after wget)
    const filePath = await promptUserInput();

    // Create variable `text` and extract from PDF
    // (Task wording references extractTextFromPDF directly; extractFromPDF calls it.)
    const text = await extractTextFromPDF(filePath);

    // Generate a single resume embedding
    const resumeEmbedding = await generateEmbeddings(text);

    // Retrieve the collection and query for top 5 similar jobs
    const collection = await chroma.getCollection({ name: collectionName });
    const results = await collection.query({
      queryEmbeddings: [resumeEmbedding],
      n: 5,
    });

    // Check results and print titles of recommended jobs
    if (results && results.ids && results.ids[0] && results.ids[0].length > 0) {
      console.log("\nTop 10 Recommended Jobs from your resume:");
      results.ids[0].forEach((id, index) => {
        const idx = parseInt(id, 10);
        const job = jobPostings[idx];
        if (job) {
          console.log(`Top ${index + 1}: ${job.jobTitle} @ ${job.company} | ${job.location} | ${job.jobType} | ${job.salary}`);
        }
      });
    } else {
      console.log("No similar jobs found.");
    }
  } catch (err) {
    console.error("Error in smartRecommendationSystem:", err);
    throw err;
  }
}

if (require.main === module) main();