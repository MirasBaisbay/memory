"""
TensorRT-LLM Client Wrapper for WALL-E
Provides OpenAI-compatible API interface for TensorRT-LLM inference
Optimized for NVIDIA Jetson Orin Nano (8GB VRAM)
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Any, Iterator, Union
from dataclasses import dataclass
from datetime import datetime

# TensorRT-LLM imports
try:
    import tensorrt_llm
    from tensorrt_llm.runtime import ModelRunner, GenerationSession
    from transformers import AutoTokenizer
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    print("⚠️ TensorRT-LLM not available. Install with: pip install tensorrt_llm")


@dataclass
class ChatCompletionChoice:
    """Mimics OpenAI ChatCompletionChoice"""
    index: int
    message: 'ChatCompletionMessage'
    finish_reason: Optional[str] = None


@dataclass
class ChatCompletionMessage:
    """Mimics OpenAI ChatCompletionMessage"""
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None


@dataclass
class ChatCompletionChunk:
    """Mimics OpenAI streaming chunk"""
    id: str
    choices: List['ChatCompletionChunkChoice']
    created: int
    model: str
    object: str = "chat.completion.chunk"


@dataclass
class ChatCompletionChunkChoice:
    """Mimics OpenAI streaming choice"""
    index: int
    delta: 'Delta'
    finish_reason: Optional[str] = None


@dataclass
class Delta:
    """Mimics OpenAI delta object"""
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None


@dataclass
class Function:
    """Tool call function"""
    name: str
    arguments: str


@dataclass
class ChatCompletionMessageToolCall:
    """Mimics OpenAI tool call"""
    id: str
    type: str
    function: Function


class TensorRTLLMClient:
    """
    TensorRT-LLM client with OpenAI-compatible interface.
    Designed for efficient inference on Jetson Orin Nano.
    """

    def __init__(
        self,
        engine_dir: str,
        tokenizer_dir: str,
        max_input_len: int = 2048,
        max_output_len: int = 512,
        max_batch_size: int = 1,
        max_beam_width: int = 1,
        enable_streaming: bool = True,
        gpu_memory_fraction: float = 0.9,
    ):
        """
        Initialize TensorRT-LLM client.

        Args:
            engine_dir: Path to TensorRT engine files
            tokenizer_dir: Path to tokenizer files
            max_input_len: Maximum input sequence length
            max_output_len: Maximum output length
            max_batch_size: Batch size (1 for Jetson)
            max_beam_width: Beam width (1 for greedy decoding)
            enable_streaming: Enable token streaming
            gpu_memory_fraction: GPU memory fraction to use
        """
        if not TENSORRT_AVAILABLE:
            raise RuntimeError("TensorRT-LLM is not installed")

        self.engine_dir = Path(engine_dir)
        self.tokenizer_dir = Path(tokenizer_dir)
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.max_batch_size = max_batch_size
        self.max_beam_width = max_beam_width
        self.enable_streaming = enable_streaming

        # Set GPU memory fraction
        torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction)

        # Load tokenizer
        print(f"🔧 Loading tokenizer from {self.tokenizer_dir}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.tokenizer_dir),
            trust_remote_code=True,
            padding_side='left'
        )

        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load TensorRT engine
        print(f"🔧 Loading TensorRT engine from {self.engine_dir}...")
        self.runner = ModelRunner.from_dir(
            engine_dir=str(self.engine_dir),
            rank=0,
            debug_mode=False,
        )

        print(f"✅ TensorRT-LLM initialized successfully")
        print(f"   Max Input: {max_input_len} | Max Output: {max_output_len}")

    def _apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        """
        Apply Qwen3 chat template to messages.

        Qwen3 format:
        <|im_start|>system
        You are a helpful assistant.<|im_end|>
        <|im_start|>user
        Hello!<|im_end|>
        <|im_start|>assistant
        """
        formatted_messages = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                formatted_messages.append(f"<|im_start|>system\n{content}<|im_end|>")
            elif role == "user":
                formatted_messages.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == "assistant":
                formatted_messages.append(f"<|im_start|>assistant\n{content}<|im_end|>")
            elif role == "tool":
                # Handle tool responses
                tool_name = msg.get("name", "tool")
                formatted_messages.append(f"<|im_start|>tool\n[{tool_name}] {content}<|im_end|>")

        # Add assistant prefix for generation
        formatted_messages.append("<|im_start|>assistant\n")

        return "\n".join(formatted_messages)

    def _extract_tool_calls(self, text: str) -> tuple[str, Optional[List[Dict]]]:
        """
        Extract tool calls from LLM response.

        Expected format:
        <tool_call>
        {"name": "function_name", "arguments": {"arg1": "value1"}}
        </tool_call>
        """
        import re

        tool_calls = []
        content_parts = []

        # Pattern to match tool calls
        pattern = r'<tool_call>(.*?)</tool_call>'
        matches = re.finditer(pattern, text, re.DOTALL)

        last_end = 0
        for i, match in enumerate(matches):
            # Add text before tool call
            content_parts.append(text[last_end:match.start()].strip())

            # Parse tool call
            try:
                tool_json = json.loads(match.group(1).strip())
                tool_calls.append({
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {
                        "name": tool_json.get("name", ""),
                        "arguments": json.dumps(tool_json.get("arguments", {}))
                    }
                })
            except json.JSONDecodeError:
                # If parsing fails, treat as regular text
                content_parts.append(match.group(0))

            last_end = match.end()

        # Add remaining text
        content_parts.append(text[last_end:].strip())

        # Combine content
        content = " ".join(p for p in content_parts if p).strip()

        return content, tool_calls if tool_calls else None

    def _generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        streaming: bool = False,
    ) -> Union[str, Iterator[str]]:
        """
        Generate text using TensorRT-LLM.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Repetition penalty
            streaming: Enable streaming output

        Returns:
            Generated text or iterator of tokens
        """
        # Tokenize input
        input_ids = self.tokenizer.encode(
            prompt,
            return_tensors='pt',
            truncation=True,
            max_length=self.max_input_len
        ).to('cuda')

        # Run inference
        with torch.no_grad():
            outputs = self.runner.generate(
                batch_input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                end_id=self.tokenizer.eos_token_id,
                pad_id=self.tokenizer.pad_token_id,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                num_beams=self.max_beam_width,
                output_sequence_lengths=True,
                return_dict=True,
                streaming=streaming,
            )

        if streaming:
            # Streaming mode - yield tokens
            def token_generator():
                for output in outputs:
                    token_text = self.tokenizer.decode(
                        output,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    )
                    yield token_text
            return token_generator()
        else:
            # Non-streaming mode
            output_ids = outputs['output_ids'][0][0]  # [batch, beam, seq_len]
            output_text = self.tokenizer.decode(
                output_ids[len(input_ids[0]):],  # Remove input tokens
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            return output_text

    def create(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        stream: bool = False,
        **kwargs
    ):
        """
        Create chat completion (OpenAI-compatible interface).

        Args:
            model: Model name (ignored, uses loaded model)
            messages: List of message dicts
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            tools: Available tools (functions)
            tool_choice: Tool selection mode
            stream: Enable streaming

        Returns:
            ChatCompletion or iterator of chunks
        """
        # Apply chat template
        prompt = self._apply_chat_template(messages)

        # Add tool descriptions if provided
        if tools:
            tool_desc = self._format_tools_for_prompt(tools)
            prompt = prompt.replace(
                "<|im_start|>assistant\n",
                f"{tool_desc}\n<|im_start|>assistant\n"
            )

        # Set max tokens
        max_new_tokens = max_tokens or self.max_output_len

        if stream:
            # Return streaming response
            return self._create_stream(
                prompt=prompt,
                model=model,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        else:
            # Non-streaming response
            output_text = self._generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                streaming=False,
            )

            # Extract tool calls if present
            content, tool_calls = self._extract_tool_calls(output_text)

            # Build response
            message = ChatCompletionMessage(
                role="assistant",
                content=content,
            )

            # Add tool calls if found
            if tool_calls:
                message.tool_calls = [
                    ChatCompletionMessageToolCall(
                        id=tc["id"],
                        type="function",
                        function=Function(
                            name=tc["function"]["name"],
                            arguments=tc["function"]["arguments"]
                        )
                    )
                    for tc in tool_calls
                ]

            # Return response
            class Response:
                def __init__(self, message):
                    self.choices = [ChatCompletionChoice(
                        index=0,
                        message=message,
                        finish_reason="stop"
                    )]

            return Response(message)

    def _create_stream(
        self,
        prompt: str,
        model: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> Iterator[ChatCompletionChunk]:
        """Create streaming chat completion"""

        # For now, implement pseudo-streaming by chunking the output
        # Full streaming requires custom TensorRT-LLM integration
        output_text = self._generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            streaming=False,
        )

        # Extract tool calls
        content, tool_calls = self._extract_tool_calls(output_text)

        # Chunk the content for streaming effect
        chunk_size = 5  # Characters per chunk
        chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]

        # First chunk with role
        yield ChatCompletionChunk(
            id="chatcmpl-" + str(datetime.now().timestamp()),
            choices=[ChatCompletionChunkChoice(
                index=0,
                delta=Delta(role="assistant", content=""),
                finish_reason=None
            )],
            created=int(datetime.now().timestamp()),
            model=model,
        )

        # Content chunks
        for chunk in chunks:
            yield ChatCompletionChunk(
                id="chatcmpl-" + str(datetime.now().timestamp()),
                choices=[ChatCompletionChunkChoice(
                    index=0,
                    delta=Delta(content=chunk),
                    finish_reason=None
                )],
                created=int(datetime.now().timestamp()),
                model=model,
            )

        # Tool calls chunk if present
        if tool_calls:
            yield ChatCompletionChunk(
                id="chatcmpl-" + str(datetime.now().timestamp()),
                choices=[ChatCompletionChunkChoice(
                    index=0,
                    delta=Delta(tool_calls=tool_calls),
                    finish_reason=None
                )],
                created=int(datetime.now().timestamp()),
                model=model,
            )

        # Final chunk
        yield ChatCompletionChunk(
            id="chatcmpl-" + str(datetime.now().timestamp()),
            choices=[ChatCompletionChunkChoice(
                index=0,
                delta=Delta(),
                finish_reason="stop"
            )],
            created=int(datetime.now().timestamp()),
            model=model,
        )

    def _format_tools_for_prompt(self, tools: List[Dict]) -> str:
        """Format tools as text for the prompt"""
        if not tools:
            return ""

        tool_descriptions = ["You have access to the following tools:"]

        for tool in tools:
            func = tool.get("function", {})
            name = func.get("name", "")
            desc = func.get("description", "")
            params = func.get("parameters", {})

            tool_desc = f"\n- {name}: {desc}"
            if params and "properties" in params:
                props = params["properties"]
                required = params.get("required", [])
                tool_desc += f"\n  Parameters: {', '.join(props.keys())}"
                if required:
                    tool_desc += f"\n  Required: {', '.join(required)}"

            tool_descriptions.append(tool_desc)

        tool_descriptions.append(
            "\nTo use a tool, respond with: <tool_call>{\"name\": \"tool_name\", \"arguments\": {\"arg\": \"value\"}}</tool_call>"
        )

        return "\n".join(tool_descriptions)


class TensorRTChatCompletions:
    """Mimics OpenAI chat.completions interface"""

    def __init__(self, client: TensorRTLLMClient):
        self.client = client

    def create(self, **kwargs):
        return self.client.create(**kwargs)


class TensorRTChat:
    """Mimics OpenAI chat interface"""

    def __init__(self, client: TensorRTLLMClient):
        self.completions = TensorRTChatCompletions(client)


class TensorRTOpenAI:
    """
    OpenAI-compatible wrapper for TensorRT-LLM.
    Drop-in replacement for OpenAI client.
    """

    def __init__(
        self,
        engine_dir: str,
        tokenizer_dir: str,
        **kwargs
    ):
        """
        Initialize TensorRT OpenAI client.

        Args:
            engine_dir: Path to TensorRT engine
            tokenizer_dir: Path to tokenizer
            **kwargs: Additional arguments for TensorRTLLMClient
        """
        self.client = TensorRTLLMClient(
            engine_dir=engine_dir,
            tokenizer_dir=tokenizer_dir,
            **kwargs
        )
        self.chat = TensorRTChat(self.client)
