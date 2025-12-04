// --- Tab Switching ---
function openTab(evt, tabName) {
    var i, tabcontent, tabbuttons;
    tabcontent = document.getElementsByClassName("tab-content");
    for (i = 0; i < tabcontent.length; i++) {
        tabcontent[i].classList.remove("active");
    }
    tabbuttons = document.getElementsByClassName("tab-button");
    for (i = 0; i < tabbuttons.length; i++) {
        tabbuttons[i].classList.remove("active");
    }
    document.getElementById(tabName).classList.add("active");
    evt.currentTarget.classList.add("active");
}

// --- Image Comparison Slider ---
document.addEventListener('DOMContentLoaded', function() {
    const slider = document.getElementById('image-slider');
    if (slider) {
        const gradcamImg = document.getElementById('gradcam-image');
        slider.addEventListener('input', function() {
            const value = this.value;
            gradcamImg.style.clipPath = `inset(0 ${100 - value}% 0 0)`;
            this.style.left = `${value}%`;
        });
    }

    // --- Chat Functionality ---
    const sendButton = document.getElementById('send-button');
    const chatInput = document.getElementById('chat-input');
    const chatBox = document.getElementById('chat-box');
    const micButton = document.getElementById('mic-button');

    if (sendButton && chatInput && chatBox) {
        const sendMessage = () => {
            const query = chatInput.value.trim();
            if (query === '') return;

            // Add user message to chat
            chatBox.innerHTML += `<div class="chat-message user-message"><p>${query}</p></div>`;
            chatInput.value = '';

            // Send to backend
            fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: query })
            })
            .then(response => response.json())
            .then(data => {
                if (data.response) {
                    chatBox.innerHTML += `<div class="chat-message ai-message"><p>${data.response.text}</p><small><em>Source: ${data.response.source}</em></small></div>`;
                    chatBox.scrollTop = chatBox.scrollHeight;
                } else {
                    chatBox.innerHTML += `<div class="chat-message ai-message"><p>Sorry, I couldn't process that.</p></div>`;
                }
            })
            .catch(error => {
                console.error('Error:', error);
                chatBox.innerHTML += `<div class="chat-message ai-message"><p>Connection error. Please try again.</p></div>`;
            });
        };

        sendButton.addEventListener('click', sendMessage);
        chatInput.addEventListener('keyup', (event) => {
            if (event.key === 'Enter') {
                sendMessage();
            }
        });
    }

    // --- Voice Input (Web Speech API) ---
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        const recognition = new SpeechRecognition();
        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.lang = 'en-US';

        if (micButton) {
            micButton.addEventListener('click', () => {
                recognition.start();
                micButton.textContent = '🔴 Listening...';
            });

            recognition.onresult = (event) => {
                const transcript = event.results[0][0].transcript;
                chatInput.value = transcript;
                micButton.textContent = '🎤';
            };

            recognition.onerror = (event) => {
                console.error('Speech recognition error:', event.error);
                micButton.textContent = '🎤';
            };
        }
    } else {
        if (micButton) micButton.style.display = 'none';
    }
});

// --- Sidebar Toggle ---
document.addEventListener("DOMContentLoaded", () => {
    const sidebar = document.querySelector('.sidebar');
    const toggleBtn = document.getElementById('sidebar-toggle');

    if (toggleBtn) {
        toggleBtn.addEventListener('click', () => {
            sidebar.classList.toggle('collapsed');
        });
    }
});

document.addEventListener("DOMContentLoaded", () => {
    const sidebar = document.querySelector('.sidebar');
    const toggleBtn = document.getElementById('sidebar-toggle');
    const openChatBtn = document.getElementById('open-chat-btn');
    const chatLink = document.getElementById('chat-link');

    console.log('DOM Loaded');
    console.log('Sidebar:', sidebar);
    console.log('Toggle Button:', toggleBtn);

    // Sidebar toggle functionality
    if (toggleBtn && sidebar) {
        toggleBtn.addEventListener('click', (e) => {
            e.preventDefault();
            console.log('Toggle button clicked');
            sidebar.classList.toggle('collapsed');
            console.log('Sidebar collapsed:', sidebar.classList.contains('collapsed'));
        });
    } else {
        console.error('Sidebar or toggle button not found!');
    }

    // Open chat button functionality
    if (openChatBtn) {
        openChatBtn.addEventListener('click', () => {
            console.log('Opening chat...');
            // Add your chat opening logic here
            // For example: window.open('/chat', '_blank');
        });
    }

    // Sync sidebar chat link with header button
    if (chatLink && openChatBtn) {
        chatLink.addEventListener('click', (e) => {
            e.preventDefault();
            openChatBtn.click();
        });
    }
});
