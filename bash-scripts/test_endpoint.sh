#!/bin/bash
# 2. Test the endpoint
curl --silent https://ai-poc.hms.edu/v1/messages \
  --header "Authorization: Bearer $HMS_AI_TOKEN" \
  --header "anthropic-version: 2023-06-01" \
  --header "Content-Type: application/json" \
  --data '{"model": "muse-glimmer", "max_tokens":256,
           "messages":[{"role":"user","content":"Say hello in one sentence."}]}'