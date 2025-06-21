# PDF Chat API - Curl Commands Reference

## Environment Variables

```bash
export BASE_URL="http://localhost:8000"
export USER_ID="demo_user"
```

## 1. Health Check

### Basic Health Check

```bash
curl -X GET "$BASE_URL/" \
  -H "accept: application/json"
```

### Detailed Health Check (Router)

```bash
curl -X GET "$BASE_URL/health" \
  -H "accept: application/json"
```

**Expected Response:**

```json
{
  "status": "healthy",
  "gemini_available": true,
  "mongodb_connected": true,
  "features": [
    "gemini_chat",
    "gemini_rag",
    "document_chunking",
    "semantic_search"
  ]
}
```

## 2. Upload PDFs

### Upload Single PDF

```bash
curl -X POST "$BASE_URL/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "user_id=$USER_ID" \
  -F "files=@/path/to/your/document.pdf"
```

### Upload Multiple PDFs

```bash
curl -X POST "$BASE_URL/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "user_id=$USER_ID" \
  -F "files=@/path/to/document1.pdf" \
  -F "files=@/path/to/document2.pdf"
```

**Expected Response:**

```json
{
  "message": "Successfully processed 1 PDF(s)",
  "files_processed": 1,
  "total_text_length": 5420
}
```

## 3. Chat Endpoints

### General Chat (No PDF Context)

```bash
curl -X POST "$BASE_URL/chat" \
  -H "Content-Type: application/json" \
  -H "accept: application/json" \
  -d '{
    "user_id": "'$USER_ID'",
    "message": "Hello! How can you help me today?"
  }'
```

### Ask About Uploaded PDF

```bash
curl -X POST "$BASE_URL/chat" \
  -H "Content-Type: application/json" \
  -H "accept: application/json" \
  -d '{
    "user_id": "'$USER_ID'",
    "message": "What is the main topic of the uploaded document?"
  }'
```

### Request Document Summary

```bash
curl -X POST "$BASE_URL/chat" \
  -H "Content-Type: application/json" \
  -H "accept: application/json" \
  -d '{
    "user_id": "'$USER_ID'",
    "message": "Can you summarize the key points from the document?"
  }'
```

### Ask Specific Questions

```bash
curl -X POST "$BASE_URL/chat" \
  -H "Content-Type: application/json" \
  -H "accept: application/json" \
  -d '{
    "user_id": "'$USER_ID'",
    "message": "What are the main conclusions mentioned in the document?"
  }'
```

**Expected Response:**

```json
{
  "user_id": "demo_user",
  "question": "What is the main topic?",
  "answer": "Based on the document content...",
  "source": "gemini_rag",
  "timestamp": "2025-06-21T10:30:00"
}
```

## 4. History and Data Retrieval

### Get Chat History

```bash
curl -X GET "$BASE_URL/history/$USER_ID" \
  -H "accept: application/json"
```

### Get Messages (Router Endpoint)

```bash
curl -X GET "$BASE_URL/messages/$USER_ID" \
  -H "accept: application/json"
```

### Get Document Statistics

```bash
curl -X GET "$BASE_URL/documents/$USER_ID" \
  -H "accept: application/json"
```

**Expected Response:**

```json
{
  "user_id": "demo_user",
  "has_documents": true,
  "total_chunks": 15,
  "total_words": 2500,
  "total_chars": 15000
}
```

## 5. Cleanup Operations

### Clear User Session (PDFs + Cache)

```bash
curl -X DELETE "$BASE_URL/session/$USER_ID" \
  -H "accept: application/json"
```

### Clear Chat Messages Only

```bash
curl -X DELETE "$BASE_URL/messages/$USER_ID" \
  -H "accept: application/json"
```

### Clear Documents Only

```bash
curl -X DELETE "$BASE_URL/documents/$USER_ID" \
  -H "accept: application/json"
```

## 6. Testing Workflows

### Complete Testing Workflow

```bash
#!/bin/bash
# PDF Chat API Testing Script

BASE_URL="http://localhost:8000"
USER_ID="test_user_$(date +%s)"

echo "🚀 Starting PDF Chat API Tests"
echo "Using USER_ID: $USER_ID"

# 1. Health Check
echo "1. Checking API health..."
curl -s "$BASE_URL/" | jq '.'

# 2. Upload a PDF
echo "2. Uploading PDF..."
curl -s -X POST "$BASE_URL/upload" \
  -F "user_id=$USER_ID" \
  -F "files=@sample.pdf" | jq '.'

# 3. Ask about the document
echo "3. Asking about document..."
curl -s -X POST "$BASE_URL/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "'$USER_ID'",
    "message": "What is this document about?"
  }' | jq '.'

# 4. Get chat history
echo "4. Getting chat history..."
curl -s "$BASE_URL/history/$USER_ID" | jq '.'

# 5. Get document stats
echo "5. Getting document stats..."
curl -s "$BASE_URL/documents/$USER_ID" | jq '.'

# 6. General chat test
echo "6. Testing general chat..."
curl -s -X POST "$BASE_URL/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "'$USER_ID'",
    "message": "What is artificial intelligence?"
  }' | jq '.'

# 7. Cleanup
echo "7. Cleaning up..."
curl -s -X DELETE "$BASE_URL/session/$USER_ID" | jq '.'

echo "✅ Tests completed!"
```

## 7. Error Testing

### Test Invalid PDF Upload

```bash
curl -X POST "$BASE_URL/upload" \
  -F "user_id=$USER_ID" \
  -F "files=@invalid_file.txt"
```

### Test Empty Message

```bash
curl -X POST "$BASE_URL/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "'$USER_ID'",
    "message": ""
  }'
```

### Test Non-existent User History

```bash
curl -X GET "$BASE_URL/history/non_existent_user"
```

## 8. Load Testing

### Simple Load Test

```bash
#!/bin/bash
# Simple load test - send 10 concurrent requests

for i in {1..10}; do
  curl -s -X POST "$BASE_URL/chat" \
    -H "Content-Type: application/json" \
    -d '{
      "user_id": "load_test_'$i'",
      "message": "Test message '$i'"
    }' &
done
wait
echo "Load test completed"
```

## 9. Response Validation

### Using jq for JSON validation

```bash
# Check if response contains required fields
curl -s "$BASE_URL/chat" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "'$USER_ID'", "message": "test"}' \
  | jq 'has("answer") and has("source") and has("timestamp")'
```

### Check API response time

```bash
curl -w "Response time: %{time_total}s\n" \
  -o /dev/null -s "$BASE_URL/"
```

## 10. Common Issues & Troubleshooting

### Check if backend is running

```bash
curl -f "$BASE_URL/" > /dev/null 2>&1 && echo "API is running" || echo "API is down"
```

### Test with verbose output

```bash
curl -v -X POST "$BASE_URL/chat" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "'$USER_ID'", "message": "test"}'
```

### Save response to file for debugging

```bash
curl -X POST "$BASE_URL/chat" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "'$USER_ID'", "message": "test"}' \
  -o response.json
```

---

## Quick Reference

| Endpoint               | Method | Purpose                |
| ---------------------- | ------ | ---------------------- |
| `/`                    | GET    | Basic health check     |
| `/health`              | GET    | Detailed health check  |
| `/upload`              | POST   | Upload PDF files       |
| `/chat`                | POST   | Send chat messages     |
| `/history/{user_id}`   | GET    | Get chat history       |
| `/messages/{user_id}`  | GET    | Get formatted messages |
| `/documents/{user_id}` | GET    | Get document stats     |
| `/session/{user_id}`   | DELETE | Clear session          |
| `/messages/{user_id}`  | DELETE | Clear messages         |
| `/documents/{user_id}` | DELETE | Clear documents        |
