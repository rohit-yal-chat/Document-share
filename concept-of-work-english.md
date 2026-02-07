# Search Benchmark Dataset - Technical Approach

> **Date:** February 2026  
> **Objective:** Generate a comprehensive search benchmark dataset from 45,000 user profiles

---

## Goal

Generate **50,000+ test cases** with realistic search queries and ranked results to evaluate search algorithm performance.

### Why 50,000 Queries?

**Calculation:**
```
8 Query Types (A-H) × 3 Difficulty Levels (Easy/Medium/Hard) = 24 unique combinations
24 combinations × ~2,100 queries each = 50,400 queries
```

**Distribution Table:**

| Query Type | Easy | Medium | Hard | Total per Type |
|------------|------|--------|------|----------------|
| **A** (Strict/Factual) | 2,100 | 2,100 | 2,100 | 6,300 |
| **B** (Semantic) | 2,100 | 2,100 | 2,100 | 6,300 |
| **C** (Natural Language) | 2,100 | 2,100 | 2,100 | 6,300 |
| **D** (Mixed Language) | 2,100 | 2,100 | 2,100 | 6,300 |
| **E** (Negative/Exclusion) | 2,100 | 2,100 | 2,100 | 6,300 |
| **F** (Short Form) | 2,100 | 2,100 | 2,100 | 6,300 |
| **G** (Colloquial/Slang) | 2,100 | 2,100 | 2,100 | 6,300 |
| **H** (Typographical Errors) | 2,100 | 2,100 | 2,100 | 6,300 |
| **TOTAL** | **16,800** | **16,800** | **16,800** | **50,400** |

**Why this distribution is optimal:**
- Each type-difficulty combination has 2,100+ samples (statistically significant)
- Balanced coverage across all query patterns
- Ensures comprehensive benchmark for search algorithm evaluation
- Every profile from 45k dataset gets represented in queries

---

## Complete Flow (4 Phases)

```
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 1: Vector Database Creation                              │
│  Convert all 45,000 profiles into searchable vector embeddings  │
│  (One-time setup - Python script)                               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 2: Query Generation                                      │
│  10 random profiles → AI generates realistic query → save JSON  │
│  (Repeat 50,000+ times)                                         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 3: Candidate Retrieval via Vector Search                 │
│  For each query, retrieve top 50 most similar profiles          │
│  (Automated - milliseconds per query)                           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 4: AI-Powered Ranking                                    │
│  50 profiles → AI ranks top 10 with justifications              │
│  (Repeat for each query)                                        │
└─────────────────────────────────────────────────────────────────┘
```

---

# Phase 1: Vector Database Creation

## Purpose
Convert all profile data into high-dimensional vector representations for fast semantic similarity search.

## Technology Stack
- **Embedding Model:** Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Database:** FAISS (Facebook AI Similarity Search)
- **Language:** Python

## Process

1. Load all 45,000 profiles from JSON
2. Extract all searchable fields from each profile:
   - Personal info (name, gender, age, languages)
   - Location (city, state, country)
   - Professional identity (occupation, industry, seniority, titles)
   - Expertise (skills, abilities, knowledge domains)
   - Work environment (activities, context)
   - Availability (engagement status, collaboration style)
   - Lifestyle (travel style, social vibe)
   - Interests (hobbies, learning goals)
   - Semantic summary and tags
3. Concatenate all fields into searchable text
4. Generate 384-dimensional embedding vector for each profile
5. Build FAISS index for efficient similarity search
6. Save index and mappings to disk

## Python Implementation

```python
import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from tqdm import tqdm

# Load profiles
with open('SyntheticProfilesPersona_Discovery_45k_profiles.json', 'r', encoding='utf-8') as f:
    profiles = json.load(f)

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

def profile_to_text(p):
    """Extract all searchable fields from profile"""
    parts = []
    
    # Personal Info
    pii = p.get('pii', {})
    parts.append(pii.get('name', ''))
    parts.append(pii.get('gender', ''))
    location = pii.get('location', {})
    parts.extend([location.get('city', ''), location.get('state', ''), 
                  location.get('neighborhood', ''), location.get('country', '')])
    
    # Demographics
    demo = p.get('demographics', {})
    parts.append(str(demo.get('age', '')))
    parts.extend(demo.get('languages', []))
    
    # Professional Identity
    prof = p.get('professional_identity', {})
    parts.append(prof.get('occupation', ''))
    parts.extend(prof.get('professional_titles', []))
    parts.extend(prof.get('alternate_titles', []))
    parts.append(prof.get('industry', ''))
    parts.append(prof.get('seniority', ''))
    parts.append(prof.get('mission_statement', ''))
    
    # Expertise
    exp = p.get('expertise_taxonomy', {})
    parts.extend(exp.get('skills', []))
    parts.extend(exp.get('abilities', []))
    parts.extend(exp.get('knowledge_domains', []))
    parts.extend(exp.get('related_occupations', []))
    
    # Work Environment
    work = p.get('work_environment_context', {})
    parts.extend(work.get('work_activities', []))
    parts.extend(work.get('work_context', []))
    
    # Discovery Intent
    disc = p.get('discovery_intent', {})
    parts.append(disc.get('engagement_status', ''))
    parts.append(disc.get('availability_pattern', ''))
    parts.append(disc.get('collaboration_style', ''))
    
    # Lifestyle
    life = p.get('lifestyle', {})
    parts.append(life.get('travel_style', ''))
    parts.append(life.get('social_vibe', ''))
    
    # Interests
    interests = p.get('interests', {})
    parts.extend(interests.get('hobbies', []))
    parts.extend(interests.get('learning_goals', []))
    
    # Search Optimization
    search = p.get('search_optimization', {})
    parts.append(search.get('semantic_summary', ''))
    parts.extend([tag.replace('#', '') for tag in search.get('tags', [])])
    
    return ' '.join([str(x) for x in parts if x])

# Create embeddings
profile_texts = [profile_to_text(p) for p in tqdm(profiles)]
embeddings = model.encode(profile_texts, show_progress_bar=True)

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings.astype('float32'))

# Save outputs
faiss.write_index(index, 'profiles_index.faiss')
np.save('embeddings.npy', embeddings)

user_ids = [p['user_id'] for p in profiles]
with open('user_ids.json', 'w') as f:
    json.dump(user_ids, f)
```

## Output Files
| File | Description |
|------|-------------|
| `profiles_index.faiss` | Vector index for similarity search |
| `embeddings.npy` | Raw embedding vectors |
| `user_ids.json` | Profile ID to index mapping |

---

# Phase 2: Query Generation

## Purpose
Generate diverse, realistic search queries that represent actual user search behavior.

## Process

1. **Sequential Profile Selection:** Take 10 profiles at a time in sequence (profiles 1-10, then 11-20, then 21-30, etc.)
2. Present complete profile data to AI model
3. AI generates one realistic search query based on profile characteristics
4. Store query with metadata (type, difficulty, source profiles)
5. Continue until all 45,000 profiles are covered
6. Repeat with different query types and difficulty levels

**Iteration Breakdown:**
```
Total Profiles: 45,000
Profiles per batch: 10
Iterations to cover all profiles: 45,000 ÷ 10 = 4,500 iterations
Query types × Difficulty levels: 8 × 3 = 24 variations
Total queries: 4,500 × ~11 queries per profile set = 50,000+ queries
```

## Query Type Classification

| Type | Name | Description | Example |
|------|------|-------------|---------|
| **A** | Strict/Factual | Exact keyword matching | "Senior Developer Bangalore" |
| **B** | Semantic | Conceptual understanding | "Someone who builds AI systems" |
| **C** | Natural Language | Conversational queries | "I need a doctor for my startup" |
| **D** | Mixed Language | Hinglish/regional mix | "Bangalore mein accha coder chahiye" |
| **E** | Negative/Exclusion | With exclusion criteria | "Designer but not in Delhi" |
| **F** | Short Form | Abbreviated queries | "Python Mumbai" |
| **G** | Colloquial/Slang | Informal language | "Coding ninja wanted" |
| **H** | Typographical Errors | Misspelled queries | "Phython develoepr" |

## Difficulty Level Classification

| Level | Description | Example |
|-------|-------------|---------|
| **Easy** | Single attribute match | "Developer in Bangalore" |
| **Medium** | Multiple attribute match | "Senior Python Developer with ML experience" |
| **Hard** | Complex semantic understanding | "High-energy person who loves hiking and can lead tech teams" |

```python
import json

with open('SyntheticProfilesPersona_Discovery_45k_profiles.json', 'r', encoding='utf-8') as f:
    all_profiles = json.load(f)

# Sequential profile selection (not random)
def get_sequential_profiles(batch_number, batch_size=10):
    """
    Get profiles sequentially in batches
    batch_number: 0, 1, 2, ... (starts from 0)
    Returns profiles from index (batch_number * batch_size) to (batch_number * batch_size + batch_size)
    """
    start_index = batch_number * batch_size
    end_index = start_index + batch_size
    
    if start_index >= len(all_profiles):
        return None  # All profiles processed
    
    return all_profiles[start_index:end_index]

def format_profile_for_ai(profile):
    """Format complete profile data for AI input"""
    p = profile
    pii = p.get('pii', {})
    location = pii.get('location', {})
    demo = p.get('demographics', {})
    prof = p.get('professional_identity', {})
    exp = p.get('expertise_taxonomy', {})
    work = p.get('work_environment_context', {})
    disc = p.get('discovery_intent', {})
    life = p.get('lifestyle', {})
    interests = p.get('interests', {})
    search = p.get('search_optimization', {})
    
    return f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PROFILE (ID: {p['user_id']})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BASIC INFO:
   • Name: {pii.get('name', 'N/A')}
   • Gender: {pii.get('gender', 'N/A')}
   • Age: {demo.get('age', 'N/A')}
   • Languages: {', '.join(demo.get('languages', []))}

LOCATION:
   • City: {location.get('city', 'N/A')}
   • State: {location.get('state', 'N/A')}
   • Country: {location.get('country', 'N/A')}

PROFESSIONAL:
   • Occupation: {prof.get('occupation', 'N/A')}
   • Industry: {prof.get('industry', 'N/A')}
   • Seniority: {prof.get('seniority', 'N/A')}
   • Titles: {', '.join(prof.get('professional_titles', []))}

EXPERTISE:
   • Skills: {', '.join(exp.get('skills', []))}
   • Abilities: {', '.join(exp.get('abilities', []))}

WORK ENVIRONMENT:
   • Activities: {', '.join(work.get('work_activities', []))}
   • Context: {', '.join(work.get('work_context', []))}

AVAILABILITY:
   • Status: {disc.get('engagement_status', 'N/A')}
   • Collaboration Style: {disc.get('collaboration_style', 'N/A')}

LIFESTYLE:
   • Travel Style: {life.get('travel_style', 'N/A')}
   • Social Vibe: {life.get('social_vibe', 'N/A')}

INTERESTS:
   • Hobbies: {', '.join(interests.get('hobbies', []))}

SUMMARY:
   {search.get('semantic_summary', 'N/A')}
"""

# Example: Process batch 0 (profiles 1-10)
batch_number = 0  # Change this for each iteration: 0, 1, 2, ... up to 4499
sample = get_sequential_profiles(batch_number, batch_size=10)

if sample:
    profiles_text = "\n".join([format_profile_for_ai(p) for p in sample])
    
    with open(f'query_generation_prompt_batch_{batch_number}.txt', 'w', encoding='utf-8') as f:
        f.write(profiles_text)
    
    print(f"Batch {batch_number}: Profiles {batch_number*10 + 1} to {batch_number*10 + 10}")
else:
    print("All profiles processed!")

# Total batches needed: 45000 / 10 = 4500 batches
```

## AI Prompt Template (Query Generation)

```
You are a Search Query Generator for a professional discovery platform.

═══════════════════════════════════════════════════════════════════════════════
YOUR TASK
═══════════════════════════════════════════════════════════════════════════════

Analyze the 10 user profiles below and generate **11 DIFFERENT search queries**.

**Requirements:**
- Generate 11 queries from this batch of 10 profiles
- Each query should be a DIFFERENT type (A-H) or difficulty level
- Cover at least 6 different query types across your 11 queries
- Include mix of Easy, Medium, and Hard difficulty levels
- Each query should feel unique and natural

**Why 11 queries per batch?**
- Total profiles: 45,000
- Profiles per batch: 10
- Total batches: 4,500
- Target queries: 50,000
- Queries per batch needed: 50,000 ÷ 4,500 ≈ 11 queries

═══════════════════════════════════════════════════════════════════════════════
QUERY TYPE SPECIFICATIONS (Use variety across your 11 queries)
═══════════════════════════════════════════════════════════════════════════════

| Type | Name | Description | Examples |
|------|------|-------------|----------|
| A | Strict/Factual | Explicit constraints (role, location, seniority) | Easy: "Java Developer in Bangalore" | Medium: "Senior Java Developer with AWS in Bangalore" | Hard: "Staff Engineer for Fintech startup in Bangalore with Rust and Blockchain expertise" |
| B | Semantic/Conceptual | Intent/skills focused without exact titles | Easy: "Backend expert" | Medium: "Someone to scale our database infrastructure" | Hard: "Visionary leader for AI transformation initiative" |
| C | Natural Language | Conversational, full-sentence queries | Easy: "I need a graphic designer" | Medium: "I am looking for a senior designer who knows Figma and lives in Mumbai" | Hard: "We are building a new crypto exchange and need a lead compliance officer who has handled regulatory audits in India before" |
| D | Hinglish/Colloquial | English-Hindi blend or local slang | Easy: "Python developer chahiye" | Medium: "Bangalore mein koi accha React dev hai kya?" | Hard: "Ek dum solid backend banda chahiye jo jaldi join kar sake" |
| E | Negative/Exclusion | Explicit exclusions in query | Easy: "Designer not in Delhi" | Medium: "Marketing Manager but NO social media focus" | Hard: "Senior Dev needed, excluding startups, and definitely not remote" |
| F | Native Daily Use | Short, efficient phrasing (telegraphic) | Easy: "Java Bangalore" | Medium: "Senior PM Fintech Delhi" | Hard: "CTO Series B" |
| G | Language Slang | Casual/Street/Buzzwords | Easy: "Coding ninja wanted" | Medium: "Growth hacker for viral loop" | Hard: "Need a wizard to crush our tech debt" |
| H | Grammatical Errors | Typos/Poor syntax (realistic mistakes) | Easy: "Phython develoepr" | Medium: "Seniar Manger for sales in dehli" | Hard: "lokking 4 exprt in mashine lurning bagalore" |

═══════════════════════════════════════════════════════════════════════════════
DIFFICULTY LEVEL DEFINITIONS
═══════════════════════════════════════════════════════════════════════════════

| Level | Definition | Matching Complexity |
|-------|------------|---------------------|
| Easy | Direct keyword overlap with profile fields; clear matches available | Single attribute match |
| Medium | Synonyms, paraphrases, or requires some interpretation (e.g., "Coder" vs "Developer") | Multiple attribute match |
| Hard | Abstract, intent-based, indirect descriptions, or ambiguous/sparse data scenarios | Complex semantic understanding |

═══════════════════════════════════════════════════════════════════════════════
PROFILE ANALYSIS REQUIREMENTS
═══════════════════════════════════════════════════════════════════════════════

When analyzing profiles, consider ALL these dimensions:
- Specific skills and abilities
- Hobbies & interests
- Work culture preferences
- Current occupation and professional titles
- Location (city, state, country)
- Seniority level
- Industry alignment
- Availability and collaboration style
- Travel preferences
- Lifestyle attributes

═══════════════════════════════════════════════════════════════════════════════
RULES - MUST FOLLOW
═══════════════════════════════════════════════════════════════════════════════

✅ DO's:
- Simulate real user imperfections (for Type H, include 5-10% typos)
- Use synonyms and paraphrasing ("Coder" vs "Developer", "Doc" vs "Doctor")
- Think like a recruiter - natural, unpredictable search patterns
- Ensure query phrasing is unique and spontaneous
- Move beyond rigid keywords to capture authentic intent
- Consider the natural, often unpredictable way people search

❌ DON'Ts:
- Don't use actual user names in queries
- Don't copy profile text verbatim
- Don't create impossible queries (only use available profile attributes)
- Don't use identical sentence patterns across queries
- Don't limit to just technical skills - consider lifestyle, hobbies, location

═══════════════════════════════════════════════════════════════════════════════
PROFILES TO ANALYZE
═══════════════════════════════════════════════════════════════════════════════

[... 10 COMPLETE PROFILES DATA HERE ...]

═══════════════════════════════════════════════════════════════════════════════
OUTPUT FORMAT (STRICT JSON ARRAY)
═══════════════════════════════════════════════════════════════════════════════

Return an array of 11 queries in this format:

[
  {"query": "<query 1>", "type": "<A-H>", "difficulty": "<Easy/Medium/Hard>"},
  {"query": "<query 2>", "type": "<A-H>", "difficulty": "<Easy/Medium/Hard>"},
  {"query": "<query 3>", "type": "<A-H>", "difficulty": "<Easy/Medium/Hard>"},
  {"query": "<query 4>", "type": "<A-H>", "difficulty": "<Easy/Medium/Hard>"},
  {"query": "<query 5>", "type": "<A-H>", "difficulty": "<Easy/Medium/Hard>"},
  {"query": "<query 6>", "type": "<A-H>", "difficulty": "<Easy/Medium/Hard>"},
  {"query": "<query 7>", "type": "<A-H>", "difficulty": "<Easy/Medium/Hard>"},
  {"query": "<query 8>", "type": "<A-H>", "difficulty": "<Easy/Medium/Hard>"},
  {"query": "<query 9>", "type": "<A-H>", "difficulty": "<Easy/Medium/Hard>"},
  {"query": "<query 10>", "type": "<A-H>", "difficulty": "<Easy/Medium/Hard>"},
  {"query": "<query 11>", "type": "<A-H>", "difficulty": "<Easy/Medium/Hard>"}
]
```

## Output Format Example

```json
[
  {"query": "Java Developer in Bangalore", "type": "A", "difficulty": "Easy"},
  {"query": "Someone who builds AI systems", "type": "B", "difficulty": "Medium"},
  {"query": "I need a senior designer who knows Figma", "type": "C", "difficulty": "Medium"},
  {"query": "Bangalore mein accha React dev hai kya?", "type": "D", "difficulty": "Medium"},
  {"query": "Marketing Manager but NO social media focus", "type": "E", "difficulty": "Medium"},
  {"query": "Senior PM Fintech Delhi", "type": "F", "difficulty": "Medium"},
  {"query": "Growth hacker for viral loop", "type": "G", "difficulty": "Medium"},
  {"query": "Seniar Manger for sales in dehli", "type": "H", "difficulty": "Medium"},
  {"query": "Staff Engineer Rust Blockchain Fintech", "type": "A", "difficulty": "Hard"},
  {"query": "Backend expert", "type": "B", "difficulty": "Easy"},
  {"query": "Python developer chahiye", "type": "D", "difficulty": "Easy"}
]
```

**Collected Output File:** `generated_queries.json` (Array of all 50,000+ queries)

---

# Phase 3: Candidate Retrieval

## Purpose
For each generated query, retrieve the 50 most semantically similar profiles from the database.

## Process

1. Load pre-built FAISS index
2. Convert query text to embedding vector
3. Perform approximate nearest neighbor search
4. Return top 50 candidates ranked by similarity

## Python Implementation

```python
import json
import faiss
from sentence_transformers import SentenceTransformer

# Load saved data
index = faiss.read_index('profiles_index.faiss')
model = SentenceTransformer('all-MiniLM-L6-v2')

with open('user_ids.json', 'r') as f:
    user_ids = json.load(f)

with open('SyntheticProfilesPersona_Discovery_45k_profiles.json', 'r', encoding='utf-8') as f:
    all_profiles = json.load(f)

profile_lookup = {p['user_id']: p for p in all_profiles}

def search_profiles(query, top_k=50):
    """Retrieve top K similar profiles for a query"""
    query_vector = model.encode([query]).astype('float32')
    distances, indices = index.search(query_vector, top_k)
    
    results = []
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        user_id = user_ids[idx]
        results.append({
            'rank': i + 1,
            'user_id': user_id,
            'distance': float(dist),
            'profile': profile_lookup[user_id]
        })
    
    return results

# Example usage
query = "Senior Python Developer in Bangalore"
results = search_profiles(query, top_k=50)

print(f"Query: '{query}'")
print(f"Found {len(results)} candidates")
for r in results[:5]:
    p = r['profile']
    print(f"  {r['rank']}. {r['user_id']} | {p['professional_identity']['occupation']}")
```

## Performance
- **Latency:** 50-100ms per query
- **Throughput:** 10-20 queries/second

---

# Phase 4: AI-Powered Ranking

## Purpose
Apply human-quality reasoning to rank candidate profiles and generate relevance justifications.

## Process

1. Take 50 candidate profiles from Phase 3
2. Present complete profile data to AI model
3. AI evaluates and ranks top 10 profiles
4. Each ranking includes detailed justification
5. Save structured result

## Ranking Criteria

| Factor | Priority | Description |
|--------|----------|-------------|
| Occupation Match | High | Direct alignment with queried role |
| Skill Relevance | High | Presence of required skills |
| Location | Medium-High | Geographic alignment |
| Seniority | Medium | Experience level match |
| Industry | Medium | Sector alignment |
| Availability | Medium | Current engagement status |
| Lifestyle Fit | Low-Medium | Cultural and personality alignment |

## AI Prompt Template

```
You are a search quality evaluator.

Query: "[QUERY_TEXT]"
Query Type: [TYPE]
Difficulty: [DIFFICULTY]

Below are 50 candidate profiles. Rank them by relevance.
Select TOP 10 profiles and assign ranks (1 = best match, 10 = least relevant among top 10).

For each profile, provide a reason explaining the ranking.

---
[... 50 COMPLETE profiles data ...]
---

Output JSON format:
{
  "test_case_id": "TC_001",
  "query_text": "...",
  "query_category": "Type X: ...",
  "difficulty_level": "...",
  "ranking_judgments": [
    {"user_id": "...", "rank": 1, "reason": "..."},
    {"user_id": "...", "rank": 2, "reason": "..."},
    ...
    {"user_id": "...", "rank": 10, "reason": "..."}
  ],
  "notes": "..."
}
```

## Output Format

```json
{
  "test_case_id": "TC_001",
  "query_text": "Senior Python Developer in Bangalore",
  "query_category": "Type A: Strict/Factual",
  "difficulty_level": "Easy",
  "ranking_judgments": [
    {
      "user_id": "u00005678",
      "rank": 1,
      "reason": "Perfect match: Senior seniority, Python expertise, Bangalore location, Technology industry"
    },
    {
      "user_id": "u00012345",
      "rank": 2,
      "reason": "Strong match: Python Developer in Bangalore, but mid-level seniority"
    },
    {
      "user_id": "u00023456",
      "rank": 3,
      "reason": "Good match: Lead level Python engineer, Bangalore, FinTech industry"
    }
  ],
  "notes": "Query tests exact match for seniority + skill + location combination."
}
```

---

# Summary

| Phase | Input | Process | Output |
|-------|-------|---------|--------|
| **Phase 1** | 45k profiles JSON | Embedding generation | Vector database |
| **Phase 2** | 10 sequential profiles | AI query generation | 1 query + metadata |
| **Phase 3** | 1 query | Vector similarity search | 50 candidate profiles |
| **Phase 4** | 50 profiles | AI relevance ranking | Top 10 ranked results |

---

## ⚠️ Infrastructure Requirements

### Computational Challenges

This project involves processing **45,000 complex JSON profiles** (135 MB raw data) through **multiple computationally intensive stages** that require **sustained high resource usage**:

| Stage | Operation | Resource Intensity |
|-------|-----------|-------------------|
| **Phase 1** | Vector Embedding Generation | Very High (CPU + RAM) |
| **Phase 1** | FAISS Index Construction | High (RAM) |
| **Phase 3** | Vector Similarity Search (50k queries) | High (RAM + CPU) |
| **Phase 2 & 4** | AI API Processing | Continuous (Network + CPU) |

### Detailed Memory Requirements

| Operation | RAM Required | Duration |
|-----------|-------------|----------|
| Load 45k profiles into memory | 4-6 GB | Continuous |
| Sentence Transformer model loaded | 2-3 GB | Continuous |
| FAISS index in memory | 2-3 GB | Continuous |
| Embedding generation buffer | 3-4 GB | During Phase 1 |
| Query processing buffer | 2-3 GB | During Phase 3 |
| **Total Peak Usage** | **15-20 GB** | Multiple phases |

### Why Consumer Laptops Cannot Handle This

| System RAM | Status | Problem |
|------------|--------|--------|
| **4 GB** | ❌ Impossible | Cannot even load the dataset |
| **8 GB** | ❌ Will Crash | Out of memory during embedding generation |
| **16 GB** | ⚠️ Unstable | System freeze, extreme slowdown, may crash |
| **32 GB** | ✅ Minimum | Barely sufficient, system still slow |
| **64 GB+** | ✅ Recommended | Smooth processing |

### Phase 3: FAISS Search Requirements

**For 50,000 vector similarity searches:**

| Metric | Value |
|--------|-------|
| Queries to process | 50,000+ |
| Time per query | 50-100 ms |
| Total search time | 42-84 minutes (just for search) |
| RAM required (continuous) | 8-10 GB |
| CPU utilization | 80-100% during search |

**System must remain dedicated during entire search process.**

### Continuous Resource Requirements

⚠️ **Critical:** This is NOT a one-time resource need. Resources are required **continuously across multiple phases:**

| Phase | Duration | RAM Needed | CPU Needed |
|-------|----------|------------|------------|
| Phase 1 (Embeddings) | 4-8 hours | 15-20 GB | 100% |
| Phase 3 (Search × 50k) | 2-4 hours | 8-10 GB | 80-100% |
| Phase 2 & 4 (AI Ranking) | 2-3 weeks | 8-10 GB | 50-70% |
| **Total Continuous** | **3-4 weeks** | **8-20 GB** | **High** |

**System will be unusable for other tasks during processing.**

### Recommended Infrastructure

For reliable, timely execution, **dedicated server infrastructure** is required:

| Requirement | Specification |
|-------------|---------------|
| RAM | 32 GB minimum, 64 GB recommended |
| CPU | 8+ cores (16 threads) |
| Storage | 100 GB SSD |
| Network | Stable, high-speed connection |
| Uptime | 24/7 for 3-4 weeks |

---

## Timeline Estimate

| Phase | Duration | Notes |
|-------|----------|-------|
| Phase 1 | 4-6 hours | One-time vector database setup |
| Phase 2-4 | 2-3 weeks | 50k+ queries with ranking |
| **Total** | **3-4 weeks** | Depends on infrastructure and API limits |

---

## Deliverables
**Target:** 50,000+ test cases

```json
{
  "test_case_id": "TC_001",
  "query_text": "Senior Python Developer in Bangalore",
  "query_category": "Type A: Strict/Factual",
  "difficulty_level": "Easy",
  "ranking_judgments": [
    {"user_id": "u00005678", "rank": 1, "reason": "..."},
    {"user_id": "u00012345", "rank": 2, "reason": "..."},
    // ... up to rank 10
  ],
  "notes": "..."
}
```
Each test case contains:
- Query text and metadata (type, difficulty)
- Top 10 ranked profiles with justifications
- Source profile references

---

*Document prepared for implementation planning and resource allocation.*
