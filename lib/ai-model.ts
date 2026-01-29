import { openai } from "@ai-sdk/openai";
import type { LanguageModelV1 } from "ai";
import { ollama } from "ollama-ai-provider";

/**
 * Returns the default lightweight model for utility tasks (moderation, naming, etc.)
 * Uses Ollama if OpenAI is not configured, enabling fully local development.
 */
export function getUtilityModel(): LanguageModelV1 {
	if (process.env.OPENAI_API_KEY) {
		return openai("gpt-4o-mini");
	}
	// Fall back to Ollama for local development
	const ollamaModel = process.env.OLLAMA_UTILITY_MODEL || "llama3.2";
	return ollama(ollamaModel);
}
