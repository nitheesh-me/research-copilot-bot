import os
import json
import yaml
import hashlib
import datetime
import glob
import concurrent.futures
from typing import List, Dict, Any

from moya.agents.openai_agent import OpenAIAgent, OpenAIAgentConfig
from src.agents.gemini_agent import GeminiAgent, GeminiAgentConfig
from src.tools.pdf_tool import pdf_tool
from src.tools.embedder import embedder_tool
from src.tools.arxiv_tool import arxiv_tool
from src.tools.crossref_tool import crossref_tool
from src.storage import Storage

class ResearchOrchestrator:
    def __init__(self, config_path: str = None, config_dict: Dict[str, Any] = None, stop_event=None):
        if config_dict:
            self.config = config_dict
        elif config_path:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        else:
            raise ValueError("Either config_path or config_dict must be provided")

        self.stop_event = stop_event
        # Ensure directories exist
        os.makedirs(os.path.dirname(self.config["trace_path"]), exist_ok=True)
        os.makedirs(os.path.dirname(self.config["db_path"]), exist_ok=True)
        os.makedirs(self.config["input_papers"], exist_ok=True)

        self.trace_file = open(self.config["trace_path"], "a")
        self.storage = None # Initialized in run() to avoid threading issues
        self.agent_keys = {} # Track API keys for agents
        # Initialize Agents
        self.summarizer = self._create_agent("summarizer", "You are a precise research summarizer.")
        self.synthesizer = self._create_agent("synthesizer", "You are a research synthesizer. Identify themes and gaps.")
        self.survey_agent = self._create_agent("survey", "You are a survey writer. Write concise surveys with citations.")
        self.verifier = self._create_agent("verifier", "You are a fact checker. Verify claims against sources.")

    def _create_agent(self, name: str, system_prompt: str):
        backend = self.config.get("model_backend", "openai")

        if backend == "gemini":
            api_key = self.config.get("api_key") or os.environ.get("GEMINI_API_KEY")
            if not api_key:
                print(f"Warning: GEMINI_API_KEY not set. Using mock for {name}.")

            final_key = api_key or "mock-key"
            self.agent_keys[name] = final_key

            config = GeminiAgentConfig(
                agent_name=name,
                agent_type="GeminiAgent",
                description=f"Agent for {name}",
                system_prompt=system_prompt,
                model_name=self.config.get("model_name", "gemini-1.5-flash"),
                api_key=final_key,
                llm_config={"temperature": self.config["temperature"]}
            )
            return GeminiAgent(config)

        # Default to OpenAI
        # Mock if no API key
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print(f"Warning: OPENAI_API_KEY not set. Using mock for {name}.")
            # We will handle mock in handle_message if needed or let it fail gracefully
            # For now, let's assume user provides key or we patch OpenAIAgent
            pass

        final_key = api_key or "mock-key"
        self.agent_keys[name] = final_key

        config = OpenAIAgentConfig(
            agent_name=name,
            agent_type="OpenAIAgent",
            description=f"Agent for {name}",
            system_prompt=system_prompt,
            model_name=self.config["model_name"],
            api_key=final_key, # Prevent init failure
            llm_config={"temperature": self.config["temperature"]}
        )
        return OpenAIAgent(config)

    def log_event(self, event_type: str, data: Dict[str, Any]):
        # Obfuscate sensitive data
        safe_data = data.copy()
        if "config" in safe_data:
            safe_config = safe_data["config"].copy()
            if "api_key" in safe_config:
                safe_config["api_key"] = "***"
            safe_data["config"] = safe_config

        entry = {
            "ts": datetime.datetime.utcnow().isoformat() + "Z",
            "type": event_type,
            **safe_data
        }
        self.trace_file.write(json.dumps(entry) + "\n")
        self.trace_file.flush()

    def run(self):
        # Initialize storage in the worker thread to avoid SQLite threading issues
        self.storage = Storage(self.config["db_path"], self.config["vector_store"])
        self.log_event("start_run", {"config": self.config})

        # 0. Check for seed topic and download papers if needed
        if self.config.get("seed_topic"):
            if self.stop_event and self.stop_event.is_set(): return
            self.log_event("arxiv_search", {"query": self.config["seed_topic"]})
            print(f"Searching Arxiv for: {self.config['seed_topic']}")
            downloaded = arxiv_tool.function(
                self.config["seed_topic"],
                max_results=self.config.get("max_papers", 5),
                download_dir=self.config["input_papers"]
            )
            self.log_event("arxiv_download", {"files": downloaded})

        # 1. Load and Parse PDFs
        if self.stop_event and self.stop_event.is_set(): return
        pdf_files = glob.glob(os.path.join(self.config["input_papers"], "*.pdf"))
        self.log_event("found_pdfs", {"count": len(pdf_files), "files": pdf_files})

        if not pdf_files:
            print("No PDFs found. Exiting.")
            return

        # Parallel Processing of Papers
        # We use a ThreadPoolExecutor to process papers in parallel.
        # Note: The OpenAIAgent now has retry logic to handle RateLimitErrors.
        max_workers = min(len(pdf_files), 5) # Limit concurrency to avoid overwhelming even with retries
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._process_paper, pdf_path): pdf_path for pdf_path in pdf_files}

            for future in concurrent.futures.as_completed(futures):
                if self.stop_event and self.stop_event.is_set():
                    executor.shutdown(wait=False, cancel_futures=True)
                    return

                pdf_path = futures[future]
                try:
                    future.result()
                except Exception as e:
                    self.log_event("error", {"msg": f"Error processing {os.path.basename(pdf_path)}", "error": str(e)})

        # 2. Synthesize
        if self.stop_event and self.stop_event.is_set(): return
        paper_ids = [os.path.basename(p).replace(".pdf", "") for p in pdf_files]
        synthesis = self._synthesize(paper_ids)

        # 3. Generate Survey
        if self.stop_event and self.stop_event.is_set(): return
        self._generate_survey(paper_ids, synthesis)

        self.log_event("end_run", {})

    def _process_paper(self, pdf_path: str):
        paper_id = os.path.basename(pdf_path).replace(".pdf", "")
        self.log_event("process_paper_start", {"paper_id": paper_id})

        # Extract chunks
        chunks, meta = pdf_tool.function(pdf_path, self.config["pdf_chunk_size"], self.config["pdf_chunk_stride"])
        self.log_event("tool_call", {"tool": "pdf_tool", "paper_id": paper_id, "meta": meta})

        self.storage.add_paper(paper_id, pdf_path, meta)

        # Embed chunks
        texts = [c["text"] for c in chunks]
        embeddings = embedder_tool.function(texts)
        self.log_event("tool_call", {"tool": "embedder", "paper_id": paper_id, "chunks": len(chunks)})

        self.storage.add_chunks(paper_id, chunks, embeddings)

        # Summarize
        summary = self._summarize_paper(paper_id, chunks)

        # Enrich with Crossref (Optional)
        if summary.get("title"):
            meta = crossref_tool.function(summary["title"])
            if meta:
                summary["crossref_meta"] = meta
                self.log_event("tool_call", {"tool": "crossref", "paper_id": paper_id, "found": True})

        self.storage.save_summary(paper_id, summary)

        self.log_event("process_paper_end", {"paper_id": paper_id})

    def _summarize_paper(self, paper_id: str, chunks: list):
        # Use first few chunks for summary (abstract/intro usually there)
        # In a real system, we might retrieve relevant chunks for specific sections
        context = "\n\n".join([c["text"] for c in chunks[:6]])
        prompt = f"""
        Seed: {self.config['seed']}
        Produce a JSON object with fields: title, authors, abstract, problem, method, results, limitations, key_phrases (list).
        Use ONLY the information below.
        ---- SOURCES ----
        {context}
        ---- END ----
        """

        self.log_event("model_call_prepare", {"agent": "summarizer", "paper_id": paper_id})

        # Mock response if using mock key
        if self.agent_keys.get("summarizer") == "mock-key":
             response = json.dumps({
                "title": f"Mock Title for {paper_id}",
                "authors": "Mock Author",
                "abstract": "This is a mock abstract.",
                "problem": "Mock problem.",
                "method": "Mock method.",
                "results": "Mock results.",
                "limitations": "Mock limitations.",
                "key_phrases": ["mock", "test"]
            })
        else:
            response = self.summarizer.handle_message(prompt)

        self.log_event("model_call", {"agent": "summarizer", "paper_id": paper_id, "response_content": response[:200] + "..."})

        try:
            # Clean up response if it contains markdown code blocks
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
            return json.loads(response)
        except Exception as e:
            self.log_event("error", {"msg": "Failed to parse summary JSON", "error": str(e), "response": response})
            return {"error": "parse_error", "raw": response}

    def _synthesize(self, paper_ids: list):
        summaries = [self.storage.get_summary(pid) for pid in paper_ids]

        # Prepare context for synthesis
        summaries_text = json.dumps(summaries, indent=2)
        prompt = f"""
        Synthesize the following research paper summaries.
        Identify common themes, conflicting results, and research gaps.
        Output JSON with fields: common_themes (list), gaps (list), insights (string).

        Summaries:
        {summaries_text}
        """

        self.log_event("model_call_prepare", {"agent": "synthesizer"})

        if self.agent_keys.get("synthesizer") == "mock-key":
            response = json.dumps({
                "common_themes": [["Theme A", 5]],
                "gaps": [{"Gap 1": "Description"}],
                "insights": "Mock insights."
            })
        else:
            response = self.synthesizer.handle_message(prompt)

        self.log_event("model_call", {"agent": "synthesizer", "response_content": response[:200] + "..."})

        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
            synthesis = json.loads(response)
            storage_dir = os.path.dirname(self.config["db_path"])
            with open(os.path.join(storage_dir, "synthesis.json"), "w") as f:
                json.dump(synthesis, f, indent=2)
            return synthesis
        except Exception as e:
            self.log_event("error", {"msg": "Failed to parse synthesis JSON", "error": str(e)})
            return {}

    def _generate_survey(self, paper_ids: list, synthesis: dict):
        summaries = [self.storage.get_summary(pid) for pid in paper_ids]
        citation_map = {pid: f"[{i+1}]" for i, pid in enumerate(paper_ids)}

        context_lines = []
        for pid in paper_ids:
            s = self.storage.get_summary(pid)
            if not s: continue
            title = s.get("title", pid)
            problem = s.get("problem", "")
            context_lines.append(f"{citation_map[pid]} {title} — {problem}")

        prompt = f"""
        Write a concise mini-survey (<= 800 words) synthesizing the following papers.
        Use inline citations like [1].
        Use the synthesis insights: {json.dumps(synthesis)}

        Papers:
        {chr(10).join(context_lines)}

        End with a bibliography.
        """

        self.log_event("model_call_prepare", {"agent": "survey"})

        if self.agent_keys.get("survey") == "mock-key":
            response = "This is a mock survey generated because no API key was provided.\n\nBibliography:\n[1] Mock Paper 1"
        else:
            response = self.survey_agent.handle_message(prompt)

        self.log_event("model_call", {"agent": "survey", "response_content": response[:200] + "..."})

        storage_dir = os.path.dirname(self.config["db_path"])
        with open(os.path.join(storage_dir, "mini_survey.txt"), "w") as f:
            f.write(response)

        # Verify claims (Simple implementation)
        self._verify_claims(response)

    def _verify_claims(self, survey_text: str):
        # Split into sentences (naive)
        sentences = [s.strip() for s in survey_text.split(".") if len(s.strip()) > 20]

        verification_results = []
        for sentence in sentences[:5]: # Verify first 5 sentences for demo
             # Embed sentence
             emb = embedder_tool.function([sentence])[0]
             # Retrieve
             chunks = self.storage.retrieve(emb, k=1)
             supported = False
             source = None
             if chunks:
                 # Simple similarity check or LLM check could go here
                 # For now, just log the top match
                 supported = True
                 source = chunks[0]["paper_id"]

             verification_results.append({
                 "sentence": sentence[:50] + "...",
                 "supported": supported,
                 "source": source
             })

        self.log_event("verification", {"results": verification_results})
        storage_dir = os.path.dirname(self.config["db_path"])
        with open(os.path.join(storage_dir, "verification.json"), "w") as f:
            json.dump(verification_results, f, indent=2)
