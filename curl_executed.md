# --- Executed all Curl Commands in Terminal ---

PS C:\Users\Prabhu\Desktop\COSMIC SOUL INTERNSHIP PROJECT\FINAL INTERN PROJECT> $env:BASE_URL = "http://localhost:8000"
PS C:\Users\Prabhu\Desktop\COSMIC SOUL INTERNSHIP PROJECT\FINAL INTERN PROJECT> $env:USER_ID = "demo_user"

PS C:\Users\Prabhu\Desktop\COSMIC SOUL INTERNSHIP PROJECT\FINAL INTERN PROJECT> Invoke-RestMethod -Method GET -Uri "$env:BASE_URL/" -Headers @{accept = "application/json"}

---

healthy Minimal RAG Chat API True True

PS C:\Users\Prabhu\Desktop\COSMIC SOUL INTERNSHIP PROJECT\FINAL INTERN PROJECT> curl.exe -X POST "$env:BASE_URL/upload" `
  -H "accept: application/json" `
  -F "user_id=$env:USER*ID" `
-F "files=@C:/Users/Prabhu/Downloads/Intern_Task*.pdf"
{"message":"Successfully processed 1 PDF(s)","files_processed":1,"total_text_length":1577}

PS C:\Users\Prabhu\Desktop\COSMIC SOUL INTERNSHIP PROJECT\FINAL INTERN PROJECT> curl -Uri "$env:BASE_URL/history/$env:USER_ID" -Method GET -Headers @{accept="application/json"}

StatusCode : 200
StatusDescription : OK
Content : {"user_id":"demo_user","database_history":[{"user_id":"demo_user",
"question":"Summarize the document.","answer":"The document outlines an intern task to build an AI-powered chat application. The appli...
RawContent : HTTP/1.1 200 OK
Content-Length: 2627
Content-Type: application/json
Date: Sat, 21 Jun 2025 11:51:06 GMT
Server: uvicorn

Forms : {}
Headers : {[Content-Length, 2627], [Content-Type, application/json], [Date,
 Sat, 21 Jun 2025 11:51:06 GMT], [Server, uvicorn]}
Images : {}
InputFields : {}
Links : {}
ParsedHtml : mshtml.HTMLDocumentClass
RawContentLength : 2627

PS C:\Users\Prabhu\Desktop\COSMIC SOUL INTERNSHIP PROJECT\FINAL INTERN PROJECT> curl -Uri "$env:BASE_URL/session/$env:USER_ID" -Method DELETE -Headers @{accept="application/json"}

StatusCode : 200
StatusDescription : OK
Content : {"message":"Session cleared for demo_user"}
RawContent : HTTP/1.1 200 OK
Content-Length: 43
Content-Type: application/json
Date: Sat, 21 Jun 2025 11:51:13 GMT
Server: uvicorn

                    {"message":"Session cleared for demo_user"}

Forms : {}
Headers : {[Content-Length, 43], [Content-Type, application/json], [Date,
 Sat, 21 Jun 2025 11:51:13 GMT], [Server, uvicorn]}
Images : {}
InputFields : {}
Links : {}
ParsedHtml : mshtml.HTMLDocumentClass
RawContentLength : 43

# --- Additional commands to complete the workflow (outputs not captured) ---

PS C:\Users\Prabhu\Desktop\COSMIC SOUL INTERNSHIP PROJECT\FINAL INTERN PROJECT> Invoke-RestMethod -Method GET -Uri ($env:BASE_URL + "/health") -Headers @{accept = "application/json"}

PS C:\Users\Prabhu\Desktop\COSMIC SOUL INTERNSHIP PROJECT\FINAL INTERN PROJECT> Invoke-RestMethod -Method GET -Uri ($env:BASE_URL + "/messages/" + $env:USER_ID) -Headers @{accept = "application/json"}

PS C:\Users\Prabhu\Desktop\COSMIC SOUL INTERNSHIP PROJECT\FINAL INTERN PROJECT> Invoke-RestMethod -Method GET -Uri ($env:BASE_URL + "/documents/" + $env:USER_ID) -Headers @{accept = "application/json"}

PS C:\Users\Prabhu\Desktop\COSMIC SOUL INTERNSHIP PROJECT\FINAL INTERN PROJECT> Invoke-RestMethod -Method DELETE -Uri ($env:BASE_URL + "/messages/" + $env:USER_ID) -Headers @{accept = "application/json"}

PS C:\Users\Prabhu\Desktop\COSMIC SOUL INTERNSHIP PROJECT\FINAL INTERN PROJECT> Invoke-RestMethod -Method DELETE -Uri ($env:BASE_URL + "/documents/" + $env:USER_ID) -Headers @{accept = "application/json"}
