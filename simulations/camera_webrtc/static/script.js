// Poll for verification results
const POLL_INTERVAL = 1000; // 1 second

let lastResultTimestamp = null;

function pollResults() {
    fetch('/api/latest_result')
        .then(response => response.json())
        .then(data => {
            const resultDisplay = document.getElementById('resultDisplay');

            if (data.result && data.timestamp !== lastResultTimestamp) {
                // New result received
                lastResultTimestamp = data.timestamp;

                // Determine if it's a success or error
                const isSuccess = data.result.toLowerCase().includes('success');
                const messageClass = isSuccess ? 'success' : 'error';

                resultDisplay.innerHTML = `
                    <p class="result-message ${messageClass}">${data.result}</p>
                `;

                // Add animation
                resultDisplay.style.animation = 'none';
                setTimeout(() => {
                    resultDisplay.style.animation = 'fadeIn 0.5s ease';
                }, 10);
            } else if (!data.result) {
                // No result available
                resultDisplay.innerHTML = `
                    <p class="no-result">Waiting for verification...</p>
                `;
            }
        })
        .catch(error => {
            console.error('Error polling results:', error);
        });
}

// Check connection status
function checkConnection() {
    const statusIndicator = document.getElementById('connectionStatus');
    const videoFeed = document.getElementById('videoFeed');

    // Check if video feed is loading
    if (videoFeed.complete && videoFeed.naturalHeight !== 0) {
        statusIndicator.textContent = 'Connected';
        statusIndicator.classList.add('connected');
    } else {
        statusIndicator.textContent = 'Connecting...';
        statusIndicator.classList.remove('connected');
    }
}

// Add CSS animation for fade in
const style = document.createElement('style');
style.textContent = `
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
`;
document.head.appendChild(style);

// Start polling when page loads
document.addEventListener('DOMContentLoaded', () => {
    // Check connection status
    checkConnection();
    setInterval(checkConnection, 5000);

    // Start polling for results
    pollResults();
    setInterval(pollResults, POLL_INTERVAL);

    console.log('WebRTC Camera System initialized');
});
