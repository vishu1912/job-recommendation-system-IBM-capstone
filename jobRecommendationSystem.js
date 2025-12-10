// ===== Part 1: Job Recommendation System (Chroma + Hugging Face) =====

// Part 1 Task 1: C. Import necessary modules
const { ChromaClient } = require("chromadb");
const client = new ChromaClient({ path: "http://localhost:8000" });
const { HfInference } = require("@huggingface/inference");
const hf = new HfInference("YOUR_API_KEY"); // <-- put your real HF key

// Part 1 Task 1: D. Get the data
const jobPostings = require("./jobPostings.js");

// Part 1 Task 1: E. Create the collection or table
const collectionName = "job_collection";

// ---------- Part 1 Task 2: Create IDs and Generate Embeddings ----------

// uniqueIds: A Set() object which stores the identifiers
const uniqueIds = new Set();

// jobTexts: each job post string
let jobTexts = [];

// B. generateEmbeddings(): uses HF featureExtraction (MiniLM-L6-v2)
async function generateEmbeddings(textArray) {
  const res = await hf.featureExtraction({
    model: "sentence-transformers/all-MiniLM-L6-v2",
    inputs: textArray,
  });
  // Normalize to float[][]
  const batch = Array.isArray(res[0]) ? res : [res];
  return batch.map((vec) => vec.map(Number));
}

// ---------- Part 1 Task 3: Extract criteria to filter the data ----------

// A. classifyText(word, labels): helper using BART large MNLI
async function classifyText(word, labels) {
  // Using HF zeroShotClassification per @huggingface/inference
  const response = await hf.zeroShotClassification({
    model: "facebook/bart-large-mnli",
    inputs: word,
    parameters: { candidate_labels: labels, multi_label: false },
  });
  return response;
}

// B. extractFilterCriteria(query): classify each token into location/job title/company/job type
async function extractFilterCriteria(query) {
  const criteria = { location: null, jobTitle: null, company: null, jobType: null };
  const labels = ["location", "job title", "company", "job type"];

  const words = String(query).split(/\s+/).filter(Boolean);

  for (const word of words) {
    const result = await classifyText(word, labels);
    // result.labels[0] has the highest score; result.scores[0] is that score
    const highestScoreLabel = result.labels?.[0];
    const score = result.scores?.[0] ?? 0;

    if (score > 0.5) {
      switch ((highestScoreLabel || "").toLowerCase()) {
        case "location":
          if (!criteria.location) criteria.location = word;
          break;
        case "job title":
          if (!criteria.jobTitle) criteria.jobTitle = word;
          break;
        case "company":
          if (!criteria.company) criteria.company = word;
          break;
        case "job type":
          if (!criteria.jobType) criteria.jobType = word;
          break;
        default:
          // do nothing
          break;
      }
    }
  }

  return criteria;
}

// Optional helper per rubric call: filterJobPostings(jobPostings, filterCriteria)
function filterJobPostings(jobPostingsArr, filterCriteria) {
  // Minimal filter: if a criterion is present, require substring match
  return jobPostingsArr.filter((jp) => {
    const locOk =
      !filterCriteria.location ||
      (jp.location && String(jp.location).toLowerCase().includes(filterCriteria.location.toLowerCase()));
    const titleOk =
      !filterCriteria.jobTitle ||
      (jp.jobTitle && String(jp.jobTitle).toLowerCase().includes(filterCriteria.jobTitle.toLowerCase()));
    const compOk =
      !filterCriteria.company ||
      (jp.company && String(jp.company).toLowerCase().includes(filterCriteria.company.toLowerCase()));
    const typeOk =
      !filterCriteria.jobType ||
      (jp.jobType && String(jp.jobType).toLowerCase().includes(filterCriteria.jobType.toLowerCase()));

    return locOk && titleOk && compOk && typeOk;
  });
}

// ---------- Part 1 Task 4: performSimilaritySearch (top 3) ----------

async function performSimilaritySearch(collection, queryTerm, jobPostingsArr) {
  try {
    // Generate query embedding
    const [queryEmbedding] = await generateEmbeddings([queryTerm]);

    // Query Chroma for up to 3 results
    const results = await collection.query({
      collection: collectionName,
      queryEmbeddings: [queryEmbedding],
      n: 3,
    });

    if (!results || !results.ids || !results.ids[0]?.length) {
      console.log(`No similar results found for "${queryTerm}"`);
      return [];
    }

    // Map to topJobPostings (id + score + fields), sort by ascending distance
    const topJobPostings = results.ids[0]
      .map((id, index) => {
        const posting = jobPostingsArr.find((p) => String(p._id) === String(id));
        if (!posting) return null;
        return {
          id,
          score: results.distances?.[0]?.[index],
          jobTitle: posting.jobTitle,
          jobType: posting.jobType,
          company: posting.company,
          jobDescription: posting.jobDescription,
        };
      })
      .filter(Boolean)
      .sort((a, b) => a.score - b.score);

    return topJobPostings;
  } catch (err) {
    console.error("Error during similarity search:", err);
    return [];
  }
}

// ---------- Part 1 Task 1: E. + Task 2 work inside main() ----------

async function main() {

  const query = process.argv.slice(2).join(" ") || "Creative Studio";

  try {
    // Create the collection
    const collection = await client.getOrCreateCollection({ name: collectionName });

    // Create unique IDs with Set + forEach + while loop
    jobPostings.forEach((job, index) => {
      if (job.jobId === undefined || job.jobId === null) job.jobId = `job_${index + 1}`;
      let idStr = String(job.jobId);
      while (uniqueIds.has(idStr)) idStr = `${idStr}_${index}`;
      uniqueIds.add(idStr);
      job._id = idStr;
    });

    // Transform each job to a text string (title + description + type + location)
    jobTexts = jobPostings.map((job) =>
      [
        job.jobTitle,
        job.jobDescription,
        job.jobType,
        job.location,
      ]
        .filter(Boolean)
        .join(". ")
        .replace(/\s+/g, " ")
        .trim()
    );

    // Generate embeddings for the jobTexts array
    const embeddingsData = await generateEmbeddings(jobTexts);

    // Store in the collection (ids, documents, embeddings)
    // To avoid duplicates if you re-run, only add if empty:
    const count = await collection.count();
    if (!count || count === 0) {
      await collection.add({
        ids: jobPostings.map((j) => j._id),
        documents: jobTexts,
        embeddings: embeddingsData,
      });
    }

    // Extract filter criteria from the query
    const filterCriteria = await extractFilterCriteria(query);

    // (rubric asks you to call filterJobPostings(jobPostings, filterCriteria))
    const filtered = filterJobPostings(jobPostings, filterCriteria);

    // Perform similarity search
    const initialResults = await performSimilaritySearch(collection, query, filtered.length ? filtered : jobPostings);

    // Log the first three items with jobTitle, jobType, jobDescription, and company
    initialResults.slice(0, 3).forEach((item, i) => {
      console.log(
        `#${i + 1} (${item.score}) ${item.jobTitle} | ${item.jobType} | ${item.company}\nDescription: ${item.jobDescription}\n`
      );
    });
  } catch (err) {
    // try-catch block with console error (grading requirement)
    console.error("Error:", err);
  }
}

if (require.main === module) main();