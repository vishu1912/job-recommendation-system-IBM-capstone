
### **📘 Job Recommendation System — IBM Capstone**

Vector Database + Hugging Face Embeddings + ChromaDB

This repository contains my IBM Vector Database Certification Capstone project:
a Job Recommendation System powered by Hugging Face embedding models and stored/retrieved using ChromaDB.

The system has two intelligent recommendation pipelines:

⸻

**🔹 Part 1 — Text Query Job Recommendation System**

Users enter a query like:

“Remote React developer role in Vancouver”
“Creative studio jobs Adobe marketing”
“Nurse position at Mayo Clinic”

The system automatically:
	1.	Extracts filters (location, job title, job type, company) via Zero-Shot Classification
	2.	Vectorizes all job postings using
sentence-transformers/all-MiniLM-L6-v2
	3.	Stores embeddings into ChromaDB
	4.	Runs vector similarity search to return the top 3 matching jobs

⸻

**🔹 Part 2 — Resume-Based Job Recommendation System (PDF)**

Users provide a PDF resume, and the system:
	1.	Extracts text using pdf-parse
	2.	Embeds the entire resume
	3.	Searches ChromaDB for top 5–10 closest job embeddings
	4.	Returns job matches tailored to the resume content

⸻

**🧠 Architecture Overview**

                  ┌────────────────────────────────┐
                  │        jobPostings.js          │
                  └────────────────────────────────┘
                              │
                              ▼
                    Text Embeddings (HF)
                              │
                              ▼
                ┌────────────────────────────┐
                │        ChromaDB (local)     │
                └────────────────────────────┘
                         ▲             ▲
                         │             │
            Query Embedding         Resume Embedding
                         │             │
                         ▼             ▼
               Top-K Similarity    Top-K Similarity
                         ▼             ▼
                    Recommendations (Jobs)


⸻

**🛠️ Tech Stack**

Component	Technology
Vector DB	ChromaDB (local server)
Embedding Models	sentence-transformers/all-MiniLM-L6-v2
Text Classification	facebook/bart-large-mnli
PDF Parsing	pdf-parse
Backend Runtime	Node.js
HF API	HuggingFace Inference endpoints


⸻

## **📦 Installation & Setup**

**1️⃣ Clone the repository**

git clone https://github.com/vishu1912/job-recommendation-system-IBM-capstone.git
cd job-recommendation-system-IBM-capstone


⸻

**2️⃣ Install dependencies**

npm install

Your package.json should include:

chromadb
@huggingface/inference
pdf-parse


⸻

**3️⃣ Start ChromaDB locally**

If using Chroma server:

chroma run --host localhost --port 8000

Verify:

http://localhost:8000


⸻

**4️⃣ Add Hugging Face API Key**

In both files:

jobrecommendationsystem.js

const hf = new HfInference("YOUR_API_KEY");

smartrecommendationsystem.js

const hf = new HfInference("YOUR_API_KEY");


⸻

⸻

**🚀 Running the Project**

⸻

**▶️ Part 1 — Query-Based Job Recommendations**

Example:

node jobrecommendationsystem.js "Remote data analyst Toronto"

Output example:

#1 (0.45) Data Analyst | Full-Time | Shopify
Description: Work with BI team...


⸻

**▶️ Part 2 — Resume PDF Job Recommendations**

Run:

node smartrecommendationsystem.js

CLI will ask:

Enter the path to your resume PDF (e.g., ./resume.pdf):

Then you will see:

Top 10 Recommended Jobs from your resume:
Top 1: Machine Learning Engineer @ IBM...
Top 2: Data Analyst @ Deloitte...


⸻

**📁 Project Structure**

📦 job-recommendation-system-IBM-capstone
 ┣ 📜 jobrecommendationsystem.js        # Part 1 logic
 ┣ 📜 smartrecommendationsystem.js      # Part 2 logic
 ┣ 📜 jobPostings.js                    # Dataset used for vector DB
 ┣ 📜 README.md
 ┗ 📜 package.json


⸻

**🧮 How Vector Search Works**

✔ Convert job descriptions → embeddings

✔ Convert user query or resume → embedding

✔ Measure cosine similarity

✔ Return closest vectors (jobs)

This allows meaningful matching even when keywords differ.

⸻

**📈 Accuracy Improvements (Future Work)**

✔ Fine-tune custom embedding model
✔ Use better filtering with entity extraction
✔ Add salary prediction model
✔ Add job clustering visualization
✔ Wrap into REST API using Express or FastAPI
✔ Add front-end UI for job search & resume upload

⸻

**🖼️ Screenshots (Add later)**

/screenshots
  ├── part1-output.png
  ├── chroma-ui.png
  └── resume-matching.png


⸻

**🏁 Final Notes**

This project demonstrates:

✔ Vector database usage
✔ Embedding generation
✔ Zero-shot classification
✔ Similarity search
✔ Resume parsing with AI
✔ Intelligent job matching

An end-to-end, production-grade recommendation engine suitable for IBM Capstone certification.

⸻
