"""Code fix module - analyze errors and provide fix guidance"""
import json
from dataclasses import dataclass
from typing import Dict, Optional, Any

from langchain_core.messages import HumanMessage, SystemMessage

from config import settings, make_chat_llm
from nodes.prompt_gen import load_template


@dataclass
class CodeFixPlan:
    """Code fix plan"""
    install_script: str
    guidance_message: str
    prompt: str
    response: str


class CodeFixer:
    """Code fixer"""
    
    def __init__(self):
        self.llm = make_chat_llm(
            temperature=0.2,
            max_tokens=settings.llm_max_tokens,
        )
    
    def plan(
        self,
        feature: Dict[str, Any],
        generated_code: str,
        error_map: Dict[str, str]
    ) -> Optional[CodeFixPlan]:
        """Analyze errors and generate a fix plan

        Args:
            feature: Feature definition
            generated_code: The generated code
            error_map: Error dict; key is sample_id, value is the error message

        Returns:
            A CodeFixPlan object, or None if one cannot be generated
        """
        if not error_map:
            return None
        
        # Load the code fix template
        template_dict = load_template("code_fix")
        template_str = template_dict.get("template", "")
        
        # Prepare the fill data
        error_summary = self._summarize_errors(error_map)
        
        fill_data = {
            "feature_name": self._escape(feature.get("name", "")),
            "feature_description": self._escape(feature.get("description", "")),
            "method": self._escape(feature.get("method", "code")),
            "error_summary": self._escape(error_summary),
            "existing_code": self._escape(generated_code or ""),
            "conda_env": self._escape(settings.conda_env),
        }
        
        # Fill the template
        try:
            filled_prompt = template_str.format(**fill_data)
        except (KeyError, ValueError):
            # If template filling fails, replace manually
            filled_prompt = template_str
            for key, value in fill_data.items():
                filled_prompt = filled_prompt.replace(f"{{{key}}}", value)
        
        # Call LLM to generate fix plan (with retry mechanism)
        print(f"\n[Code Fix] Analyzing errors for feature '{feature.get('name')}'...")
        print(f"  Error count: {len(error_map)}")
        
        messages = [
            HumanMessage(content=filled_prompt)
        ]
        try:
            response = self.llm.invoke(messages)
            response_text = response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            print(f"  ❌ LLM API call failed: {type(e).__name__}: {str(e)}")
            raise
        
        # Extract JSON
        data = self._extract_json(response_text)
        if not data:
            print(f"  ⚠️  Failed to extract valid JSON from response")
            return None
        
        install_script = data.get("install_script", "").strip()
        guidance_message = data.get("guidance_message", "").strip()
        
        if not install_script and not guidance_message:
            print(f"  ⚠️  Fix plan is empty")
            return None
        
        print(f"  ✅ Generated fix plan")
        if install_script:
            print(f"    Install script: {len(install_script)} characters")
        if guidance_message:
            print(f"    Guidance message: {len(guidance_message)} characters")
        
        return CodeFixPlan(
            install_script=install_script,
            guidance_message=guidance_message,
            prompt=filled_prompt,
            response=response_text
        )
    
    def _summarize_errors(self, errors: Dict[str, str]) -> str:
        """Summarize error information"""
        seen = []
        for sid, msg in errors.items():
            summary = f"{sid}: {msg}"
            if summary not in seen:
                seen.append(summary)
            if len(seen) >= 5:  # Show at most 5 errors
                break
        return "\n".join(seen)
    
    def _extract_json(self, raw: str) -> Dict[str, Any]:
        """Extract a JSON object from text"""
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end < 0 or start >= end:
            return {}
        try:
            return json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            return {}
    
    def _escape(self, text: str) -> str:
        """Escape curly braces to avoid errors during format"""
        return text.replace("{", "{{").replace("}", "}}")
