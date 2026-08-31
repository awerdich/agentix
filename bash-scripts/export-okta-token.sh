#!/bin/bash
export HMS_AI_TOKEN="$(./get-okta-token.sh | awk -F'Token: ' '/Token:/{print $2}')"