// Results page - Chat functionality with OpenAI integration

document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const chatMessages = document.getElementById('chat-messages');
    const chatInput = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');
    const voiceBtn = document.getElementById('voice-btn');
    const chatLoading = document.getElementById('chat-loading');
    const questionBtns = document.querySelectorAll('.question-btn');
    const tabBtns = document.querySelectorAll('.tab-btn');

    // Get scan data from HTML data attributes
    const scanDataEl = document.getElementById('scan-data');
    const resultData = {
        prediction: scanDataEl ? scanDataEl.getAttribute('data-prediction') : '',
        confidence: scanDataEl ? parseFloat(scanDataEl.getAttribute('data-confidence')) : 0,
        riskLevel: scanDataEl ? scanDataEl.getAttribute('data-risk-level') : '',
        resultId: scanDataEl ? parseInt(scanDataEl.getAttribute('data-result-id')) : 0
    };

    // Make it available globally if needed
    window.resultData = resultData;

    // Conversation history
    let conversationHistory = [];

    // Initialize with result context
    if (resultData.prediction) {
        conversationHistory.push({
            role: 'system',
            content: `You are a helpful dermatology assistant. The patient has uploaded an image that was analyzed as: ${resultData.prediction} with ${(resultData.confidence * 100).toFixed(2)}% confidence and ${resultData.riskLevel} risk level. Provide helpful, empathetic responses about skin health, but always remind them to consult a dermatologist for definitive diagnosis.`
        });
    }

    // ============================================
    // TAB SWITCHING
    // ============================================
    tabBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            // Remove active class from all tabs
            tabBtns.forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            
            // Add active class to clicked tab
            this.classList.add('active');
            const tabId = this.getAttribute('data-tab');
            document.getElementById(tabId).classList.add('active');
        });
    });

    // ============================================
    // SEND MESSAGE FUNCTION
    // ============================================
    async function sendMessage(message) {
        if (!message.trim()) return;

        // Add user message to chat
        addMessage(message, 'user');
        
        // Clear input
        chatInput.value = '';
        
        // Show loading
        chatLoading.style.display = 'block';
        sendBtn.disabled = true;

        try {
            // Send to backend
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    conversation_history: conversationHistory,
                    result_id: window.resultData ? window.resultData.resultId : null
                })
            });

            const data = await response.json();

            // Hide loading
            chatLoading.style.display = 'none';
            sendBtn.disabled = false;

            if (data.success) {
                // Add AI response to chat
                addMessage(data.response, 'ai');
                
                // Update conversation history
                conversationHistory = data.conversation_history;
            } else {
                // Show error
                addMessage('Sorry, I encountered an error. Please try again.', 'ai');
                console.error('Chat error:', data.error);
            }

        } catch (error) {
            console.error('Error sending message:', error);
            chatLoading.style.display = 'none';
            sendBtn.disabled = false;
            addMessage('Sorry, I could not connect to the server. Please try again.', 'ai');
        }
    }

    // ============================================
    // ADD MESSAGE TO CHAT
    // ============================================
    function addMessage(text, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;

        const avatarDiv = document.createElement('div');
        avatarDiv.className = 'message-avatar';
        avatarDiv.innerHTML = sender === 'ai' ? '<i class="fas fa-robot"></i>' : '<i class="fas fa-user"></i>';

        const bubbleDiv = document.createElement('div');
        bubbleDiv.className = 'message-bubble';
        bubbleDiv.innerHTML = `<p>${text}</p>`;

        messageDiv.appendChild(avatarDiv);
        messageDiv.appendChild(bubbleDiv);
        
        chatMessages.appendChild(messageDiv);
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // ============================================
    // EVENT LISTENERS
    // ============================================

    // Send button click
    sendBtn.addEventListener('click', function() {
        const message = chatInput.value;
        sendMessage(message);
    });

    // Enter key in input
    chatInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            const message = chatInput.value;
            sendMessage(message);
        }
    });

    // Common question buttons
    questionBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            const question = this.getAttribute('data-question');
            chatInput.value = question;
            sendMessage(question);
        });
    });

    // ============================================
    // VOICE INPUT (Web Speech API)
    // ============================================
    let recognition = null;
    
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        recognition = new SpeechRecognition();
        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.lang = 'en-US';

        recognition.onstart = function() {
            voiceBtn.classList.add('recording');
            voiceBtn.innerHTML = '<i class="fas fa-stop"></i>';
        };

        recognition.onresult = function(event) {
            const transcript = event.results[0][0].transcript;
            chatInput.value = transcript;
            voiceBtn.classList.remove('recording');
            voiceBtn.innerHTML = '<i class="fas fa-microphone"></i>';
        };

        recognition.onerror = function(event) {
            console.error('Speech recognition error:', event.error);
            voiceBtn.classList.remove('recording');
            voiceBtn.innerHTML = '<i class="fas fa-microphone"></i>';
        };

        recognition.onend = function() {
            voiceBtn.classList.remove('recording');
            voiceBtn.innerHTML = '<i class="fas fa-microphone"></i>';
        };

        voiceBtn.addEventListener('click', function() {
            if (voiceBtn.classList.contains('recording')) {
                recognition.stop();
            } else {
                recognition.start();
            }
        });
    } else {
        // Browser doesn't support speech recognition
        voiceBtn.style.display = 'none';
        console.log('Speech recognition not supported');
    }

    // ============================================
    // ACTION BUTTONS
    // ============================================

    // Schedule with Dermatologist
    const scheduleBtn = document.querySelector('.btn-schedule');
    if (scheduleBtn) {
        scheduleBtn.addEventListener('click', function() {
            alert('Schedule feature coming soon! This will integrate with your calendar and dermatologist booking system.');
        });
    }

    // Download Report
    const downloadBtn = document.querySelector('.btn-download');
    if (downloadBtn) {
        downloadBtn.addEventListener('click', function() {
            // Generate PDF report
            if (resultData.resultId) {
                window.location.href = `/download-report/${resultData.resultId}`;
            }
        });
    }
});