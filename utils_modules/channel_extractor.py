"""Channel name extraction utility - automatically extract channel information from the dataset description"""
import re
from typing import List, Optional, Dict, Any
from config import settings, make_chat_llm


def extract_channel_names_from_description(description: str, num_channels: int) -> List[str]:
    """Extract channel names from the dataset description
    
    Args:
        description: dataset description text
        num_channels: number of channels
        
    Returns:
        List of channel names
    """
    if not description or num_channels <= 0:
        return [f"Channel {i+1}" for i in range(num_channels)]
    
    # Try to extract channel information using an LLM
    try:
        llm = make_chat_llm(
            temperature=0,
            max_tokens=500,
        )
        
        prompt = f"""Extract channel name information from the following dataset description.

Dataset description:
{description[:2000]}

Number of channels: {num_channels}

Please extract the name of each channel. If the description explicitly mentions channel names (such as w1, w2, channel 1, channel 2, etc.), extract those names.
If the description mentions stains or markers (such as Hoechst, DAPI, Phalloidin, etc.), map these to the channels.

Please return the result in JSON format, as follows:
{{
  "channels": ["Channel 1 name", "Channel 2 name", ...]
}}

If it cannot be extracted from the description, return an empty array []. Return only JSON, with no other text."""
        
        from langchain_core.messages import HumanMessage
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content.strip()
        
        # Try to parse the JSON
        import json
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(0))
            channels = result.get("channels", [])
            if len(channels) == num_channels:
                return channels
    except Exception as e:
        print(f"  [Channel Extractor] ⚠️  LLM channel name extraction failed: {e}")
    
    # Fallback: try simple keyword matching
    desc_lower = description.lower()
    channel_names = []
    
    # Check whether it contains common channel naming patterns
    # Do not hardcode channel names for specific datasets here; instead look for general patterns
    channel_patterns = [
        r'channel\s*(\d+)[\s:]+([^\n,]+)',
        r'w\s*(\d+)[\s:]+([^\n,]+)',
    ]
    
    found_channels = {}
    for pattern in channel_patterns:
        matches = re.finditer(pattern, desc_lower, re.IGNORECASE)
        for match in matches:
            idx = int(match.group(1)) - 1
            name = match.group(2).strip()
            if 0 <= idx < num_channels:
                found_channels[idx] = name
    
    # Build the channel name list
    for i in range(num_channels):
        if i in found_channels:
            channel_names.append(found_channels[i])
        else:
            channel_names.append(f"Channel {i+1}")
    
    return channel_names[:num_channels]


def extract_channel_keywords_from_description(description: str) -> List[str]:
    """Extract channel-related keywords from the dataset description (used for prompt generation)
    
    Args:
        description: dataset description text
        
    Returns:
        List of keywords
    """
    if not description:
        return []
    
    keywords = []
    desc_lower = description.lower()
    
    # Look for channel number patterns
    if re.search(r'\bw\d+\b|\bchannel\s*\d+\b', desc_lower):
        keywords.extend(["channel"])
    
    # Look for stain/marker keywords (without hardcoding specific names)
    # Look for common biological marker patterns
    marker_patterns = [
        r'\b(dapi|hoechst|phalloidin|cona|syto|mitotracker|wga)\b',
        r'\b(stain|fluorescence|marker)\b',
    ]
    
    for pattern in marker_patterns:
        if re.search(pattern, desc_lower):
            keywords.extend(["stain", "fluorescence"])
            break
    
    return list(set(keywords))  # Deduplicate
