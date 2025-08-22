package main

import (
	"context"
	"fmt"
	"strings"

	llamastackclient "github.com/llamastack/llama-stack-client-go"
	"github.com/llamastack/llama-stack-client-go/option"
)

func main() {
	// Create a new LlamaStack client configured for local instance
	client := llamastackclient.NewClient(
		option.WithBaseURL("http://localhost:8321"),
	)

	fmt.Println("=== LlamaStack RAG Demo ===\n")

	// Run the RAG demo
	if err := runRAGDemo(&client); err != nil {
		fmt.Printf("RAG Demo failed: %v\n", err)
		return
	}

	fmt.Println("\n=== Demo Complete! ===")
}

// runRAGDemo demonstrates a complete RAG pipeline using LlamaStack
func runRAGDemo(client *llamastackclient.Client) error {
	ctx := context.Background()

	// Step 1: Create a vector store
	fmt.Println("=== Step 1: Creating Vector Store ===")
	vectorStore, err := client.VectorStores.New(ctx, llamastackclient.VectorStoreNewParams{
		Name: llamastackclient.String("my-rag-store"),
	})
	if err != nil {
		return fmt.Errorf("error creating vector store: %v", err)
	}
	fmt.Printf("Created vector store: %s (ID: %s)\n", vectorStore.Name, vectorStore.ID)

	// Step 2: Upload a sample text file to the Files service first
	fmt.Println("\n=== Step 2: Uploading Sample File ===")

	// Create sample content
	sampleContent := `Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines that work and react like humans. 
	
Machine Learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed. 
	
Deep Learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data.

Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers and human language, enabling machines to understand, interpret, and generate human language.

Computer Vision is another AI field that enables computers to interpret and understand visual information from the world, such as images and videos.

These technologies are transforming industries like healthcare, finance, transportation, and entertainment by automating tasks, improving decision-making, and creating new capabilities.`

	// Create a file reader from the sample content
	fileReader := strings.NewReader(sampleContent)

	// First upload file to the Files service
	file, err := client.Files.New(ctx, llamastackclient.FileNewParams{
		File:    llamastackclient.NewFile(fileReader, "ai_concepts.txt", "text/plain"),
		Purpose: llamastackclient.FileNewParamsPurposeAssistants,
	})
	if err != nil {
		return fmt.Errorf("error uploading file: %v", err)
	}
	fmt.Printf("Uploaded file: %s (ID: %s)\n", file.Filename, file.ID)

	// Step 3: Attach the file to the vector store
	fmt.Println("\n=== Step 3: Attaching File to Vector Store ===")

	// Attach file to vector store
	_, err = client.VectorStores.Files.New(ctx, vectorStore.ID, llamastackclient.VectorStoreFileNewParams{
		FileID: file.ID,
	})
	if err != nil {
		return fmt.Errorf("error attaching file to vector store: %v", err)
	}
	fmt.Printf("File attached to vector store successfully\n")

	// Wait a moment for processing
	fmt.Println("Waiting for file processing to complete...")
	// In a real application, you might want to poll the file status

	// Step 4: Run a query against the vector store
	fmt.Println("\n=== Step 4: Running RAG Query ===")

	query := "What is machine learning and how does it relate to AI?"
	fmt.Printf("Query: %s\n", query)

	// Search the vector store
	searchResults, err := client.VectorStores.Search(ctx, vectorStore.ID, llamastackclient.VectorStoreSearchParams{
		Query: llamastackclient.VectorStoreSearchParamsQueryUnion{
			OfString: llamastackclient.String(query),
		},
		MaxNumResults: llamastackclient.Int(3), // Get top 3 results
	})
	if err != nil {
		return fmt.Errorf("error searching vector store: %v", err)
	}

	fmt.Printf("\nFound %d relevant chunks:\n", len(searchResults.Data))
	for i, result := range searchResults.Data {
		fmt.Printf("\n--- Chunk %d ---\n", i+1)
		fmt.Printf("Content: %s\n", result.Content)
		fmt.Printf("Score: %.4f\n", result.Score)
	}

	// Step 5: Use the retrieved context to generate a better answer
	fmt.Println("\n=== Step 5: Generating Enhanced Answer ===")

	// Combine retrieved chunks into context
	var contextBuilder strings.Builder
	contextBuilder.WriteString("Based on the following information:\n\n")
	for i, result := range searchResults.Data {
		contextBuilder.WriteString(fmt.Sprintf("%d. %s\n", i+1, result.Content))
	}
	contextBuilder.WriteString("\nPlease answer the question: " + query)

	// Get available LLM models for generation
	models, err := client.Models.List(ctx)
	if err != nil {
		return fmt.Errorf("error fetching models: %v", err)
	}

	var llmModel string
	for _, model := range *models {
		if model.ModelType == "llm" {
			llmModel = model.Identifier
			break
		}
	}

	if llmModel == "" {
		return fmt.Errorf("no LLM model available for generation")
	}

	// Generate answer using the retrieved context
	response, err := client.Chat.Completions.New(ctx, llamastackclient.ChatCompletionNewParams{
		Messages: []llamastackclient.ChatCompletionNewParamsMessageUnion{
			{
				OfSystem: &llamastackclient.ChatCompletionNewParamsMessageSystem{
					Content: llamastackclient.ChatCompletionNewParamsMessageSystemContentUnion{
						OfString: llamastackclient.String("You are a helpful AI assistant. Use the provided context to answer questions accurately and concisely."),
					},
				},
			},
			{
				OfUser: &llamastackclient.ChatCompletionNewParamsMessageUser{
					Content: llamastackclient.ChatCompletionNewParamsMessageUserContentUnion{
						OfString: llamastackclient.String(contextBuilder.String()),
					},
				},
			},
		},
		Model:     llmModel,
		MaxTokens: llamastackclient.Int(300),
	})

	if err != nil {
		return fmt.Errorf("error generating answer: %v", err)
	}

	// Display the generated answer
	if openAIResponse := response.AsOpenAIChatCompletion(); openAIResponse.Choices != nil {
		if len(openAIResponse.Choices) > 0 {
			message := openAIResponse.Choices[0].Message
			if assistantMessage := message.AsAssistant(); assistantMessage.Content.OfString != "" {
				fmt.Printf("\nEnhanced Answer:\n%s\n", assistantMessage.Content.OfString)
			}
		}
	}

	return nil
}
