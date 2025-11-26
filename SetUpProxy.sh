export USER="spelerin797"
export PROXY_ADDRESS="proxy.univ-tln.fr:3128"

# --- test_proxy: tests the current proxy settings using environment variables ---
test_proxy() {
  local TEST_URL="https://httpbin.org/ip"
  local VERBOSE=false

  # Parse arguments
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -v|--verbose)
        VERBOSE=true
        shift
        ;;
      *)
        echo "Unknown argument: $1"
        echo "Usage: test_proxy [--verbose|-v]"
        return 1
        ;;
    esac
  done

  local http_code curl_exit

  if $VERBOSE; then
    echo "🧪 Testing proxy connection to $TEST_URL ..."
  fi

  http_code=$(curl -sS -o /dev/null -w "%{http_code}" --max-time 10 "$TEST_URL" 2>/dev/null)
  curl_exit=$?

  if [[ -n "$http_code" && "$http_code" =~ ^(2|3) ]]; then
    if $VERBOSE; then
      echo "✅ Proxy test successful (HTTP $http_code)"
    fi
    return 0
  else
    if $VERBOSE; then
      echo "❌ Proxy test failed."
      if [[ -n "$curl_exit" ]]; then
        echo " • curl exited with code $curl_exit"
      fi
      if [[ -n "$http_code" ]]; then
        echo " • HTTP status: $http_code"
      else
        echo " • No response received from $TEST_URL."
      fi
      echo " • Possible issues: wrong password, proxy unreachable, or bad config."
    fi
    return 1
  fi
}

try_setup_proxy() {
    # Prompt for the password (hidden input)
    read -rsp "Enter proxy password: " password
    echo
    
    # URL-encode the password safely
    encoded_password=$(python3 -c "import urllib.parse;print(urllib.parse.quote('''$password'''))")

    # Construct the proxy URL
    local proxy_url="http://${USER}:${encoded_password}@${PROXY_ADDRESS}"

    # Export proxy environment variables
    export http_proxy="$proxy_url"
    export https_proxy="$proxy_url"
    export HTTP_PROXY="$proxy_url"
    export HTTPS_PROXY="$proxy_url"

    # Common local exceptions (customize if needed)
    export no_proxy="localhost,127.0.0.1,.univ-tln.fr"
    export NO_PROXY="$no_proxy"
}

proxy() {
  echo "🔍 Checking if proxy works..."
  until test_proxy; do
    echo "❌ Proxy test failed. Retrying..."
    if ! try_setup_proxy; then
      sleep 1
    fi
  done
  echo "✅ Proxy is working correctly!"
}