#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Face Recognition System...${NC}"

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Kill existing processes on ports 5001 and 3000
echo -e "${YELLOW}Killing existing processes on ports 5001 and 3000...${NC}"

# Kill Flask backend on port 5001
if lsof -ti:5001 > /dev/null 2>&1; then
    echo -e "${YELLOW}Killing process on port 5001 (Flask backend)...${NC}"
    lsof -ti:5001 | xargs kill -9 2>/dev/null
    sleep 1
fi

# Kill React frontend on port 3000
if lsof -ti:3000 > /dev/null 2>&1; then
    echo -e "${YELLOW}Killing process on port 3000 (React frontend)...${NC}"
    lsof -ti:3000 | xargs kill -9 2>/dev/null
    sleep 1
fi

echo -e "${GREEN}Ports cleared!${NC}"

# Check if frontend node_modules exists
if [ ! -d "frontend/node_modules" ]; then
    echo -e "${YELLOW}Installing React dependencies...${NC}"
    cd frontend
    npm install
    cd ..
fi

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}Shutting down servers...${NC}"
    kill $FLASK_PID $REACT_PID 2>/dev/null
    lsof -ti:5001 | xargs kill -9 2>/dev/null
    lsof -ti:3000 | xargs kill -9 2>/dev/null
    exit
}

# Trap Ctrl+C
trap cleanup INT TERM

# Start Flask backend
echo -e "${GREEN}Starting Flask backend on port 5001...${NC}"
# Try python3 first, fallback to python
if command -v python3 &> /dev/null; then
    python3 pytorch_app.py > flask.log 2>&1 &
else
    python pytorch_app.py > flask.log 2>&1 &
fi
FLASK_PID=$!

# Wait for Flask to start and verify it's running
echo -e "${YELLOW}Waiting for Flask backend to start...${NC}"
for i in {1..10}; do
    if curl -s http://localhost:5001/model_info > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Flask backend is running!${NC}"
        break
    fi
    if [ $i -eq 10 ]; then
        echo -e "${RED}✗ Flask backend failed to start. Check flask.log${NC}"
        exit 1
    fi
    sleep 1
done

# Start React frontend
echo -e "${GREEN}Starting React frontend on port 3000...${NC}"
cd frontend
# Fix for react-scripts 5.0.1 webpack dev server issue
DANGEROUSLY_DISABLE_HOST_CHECK=true BROWSER=none npm start > ../react.log 2>&1 &
REACT_PID=$!
cd ..

# Wait a bit for React to start
echo -e "${YELLOW}Waiting for React frontend to start (this may take 30-60 seconds)...${NC}"
sleep 5

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✅ Both servers are running!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Flask Backend:  http://localhost:5001${NC}"
echo -e "${GREEN}React Frontend: http://localhost:3000${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "${YELLOW}Logs:${NC}"
echo -e "  Backend:  flask.log"
echo -e "  Frontend: react.log"
echo -e "${YELLOW}Press Ctrl+C to stop both servers${NC}"

# Wait for both processes
wait $FLASK_PID $REACT_PID

