// AR-Agent Medical Application JavaScript

class ARAgent {
    constructor() {
        this.video = null;
        this.canvas = null;
        this.stream = null;
        this.isAnalyzing = false;
        this.currentSection = 'camera';
        
        this.init();
    }

    init() {
        this.setupElements();
        this.setupEventListeners();
        this.checkSystemStatus();
        this.setupNavigation();
        this.setupUpload();
    }

    setupElements() {
        this.video = document.getElementById('video');
        this.canvas = document.getElementById('canvas');
        this.startCameraBtn = document.getElementById('start-camera');
        this.captureBtn = document.getElementById('capture');
        this.stopCameraBtn = document.getElementById('stop-camera');
        this.analysisResults = document.getElementById('analysis-results');
        this.loadingOverlay = document.getElementById('loading-overlay');
        this.notification = document.getElementById('notification');
    }

    setupEventListeners() {
        // Camera controls
        this.startCameraBtn?.addEventListener('click', () => this.startCamera());
        this.captureBtn?.addEventListener('click', () => this.captureAndAnalyze());
        this.stopCameraBtn?.addEventListener('click', () => this.stopCamera());
        
        // Upload controls
        const fileInput = document.getElementById('file-input');
        const uploadArea = document.getElementById('upload-area');
        const analyzeUploadBtn = document.getElementById('analyze-upload');
        const clearUploadBtn = document.getElementById('clear-upload');
        
        fileInput?.addEventListener('change', (e) => this.handleFileSelect(e));
        uploadArea?.addEventListener('click', () => fileInput?.click());
        analyzeUploadBtn?.addEventListener('click', () => this.analyzeUploadedImage());
        clearUploadBtn?.addEventListener('click', () => this.clearUpload());
        
        // Drag and drop
        uploadArea?.addEventListener('dragover', (e) => this.handleDragOver(e));
        uploadArea?.addEventListener('dragleave', (e) => this.handleDragLeave(e));
        uploadArea?.addEventListener('drop', (e) => this.handleDrop(e));
        
        // AR Interface
        const launchARBtn = document.getElementById('launch-ar');
        launchARBtn?.addEventListener('click', () => this.launchARInterface());
    }

    setupNavigation() {
        const navLinks = document.querySelectorAll('.nav-link');
        const sections = document.querySelectorAll('.content-section');
        
        navLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const targetSection = link.dataset.section;
                
                // Update active nav link
                navLinks.forEach(l => l.classList.remove('active'));
                link.classList.add('active');
                
                // Show target section
                sections.forEach(s => s.classList.remove('active'));
                const targetElement = document.getElementById(`${targetSection}-section`);
                if (targetElement) {
                    targetElement.classList.add('active');
                    this.currentSection = targetSection;
                }
            });
        });
    }

    setupUpload() {
        // Prevent default drag behaviors
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            document.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
            });
        });
    }

    async checkSystemStatus() {
        try {
            const response = await fetch('/health');
            const status = await response.json();
            
            document.getElementById('model-status').textContent = 
                status.model_loaded ? 'Ready' : 'Loading...';
            document.getElementById('device-status').textContent = status.device || 'Unknown';
            document.getElementById('ar-status').textContent = 
                this.checkARSupport() ? 'Available' : 'Not Available';
                
        } catch (error) {
            console.error('Failed to check system status:', error);
            this.showNotification('Failed to connect to server', 'error');
        }
    }

    checkARSupport() {
        // Check for WebXR support
        return 'xr' in navigator && 'requestSession' in navigator.xr;
    }

    async startCamera() {
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    facingMode: 'environment' // Use back camera if available
                }
            });
            
            this.video.srcObject = this.stream;
            
            // Update button states
            this.startCameraBtn.disabled = true;
            this.captureBtn.disabled = false;
            this.stopCameraBtn.disabled = false;
            
            this.showNotification('Camera started successfully', 'success');
            
        } catch (error) {
            console.error('Error starting camera:', error);
            this.showNotification('Failed to start camera. Please check permissions.', 'error');
        }
    }

    stopCamera() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
            this.video.srcObject = null;
        }
        
        // Update button states
        this.startCameraBtn.disabled = false;
        this.captureBtn.disabled = true;
        this.stopCameraBtn.disabled = true;
        
        this.showNotification('Camera stopped', 'info');
    }

    async captureAndAnalyze() {
        if (!this.stream || this.isAnalyzing) return;
        
        try {
            this.isAnalyzing = true;
            this.showLoading(true);
            
            // Capture image from video
            const canvas = this.canvas;
            const context = canvas.getContext('2d');
            canvas.width = this.video.videoWidth;
            canvas.height = this.video.videoHeight;
            context.drawImage(this.video, 0, 0);
            
            // Convert to base64
            const imageData = canvas.toDataURL('image/jpeg', 0.8).split(',')[1];
            
            // Send for analysis
            const result = await this.analyzeImage(imageData);
            this.displayAnalysisResults(result);
            
        } catch (error) {
            console.error('Error capturing and analyzing:', error);
            this.showNotification('Analysis failed. Please try again.', 'error');
        } finally {
            this.isAnalyzing = false;
            this.showLoading(false);
        }
    }

    async analyzeImage(imageData, prompt = null) {
        const payload = {
            image: imageData,
            prompt: prompt || "Analyze this medical image and provide a detailed description including any notable findings, anatomical structures, and potential abnormalities."
        };
        
        const response = await fetch('/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });
        
        if (!response.ok) {
            throw new Error(`Analysis failed: ${response.statusText}`);
        }
        
        return await response.json();
    }

    displayAnalysisResults(result) {
        if (result.error) {
            this.analysisResults.innerHTML = `
                <div class="error-message">
                    <i class="fas fa-exclamation-triangle"></i>
                    <p>Analysis Error: ${result.error}</p>
                </div>
            `;
            return;
        }
        
        const html = `
            <div class="analysis-result">
                <div class="result-section">
                    <h4><i class="fas fa-eye"></i> Description</h4>
                    <p>${result.description || 'No description available'}</p>
                </div>
                
                ${result.findings ? `
                    <div class="result-section">
                        <h4><i class="fas fa-list"></i> Key Findings</h4>
                        <ul>
                            ${result.findings.map(finding => `<li>${finding}</li>`).join('')}
                        </ul>
                    </div>
                ` : ''}
                
                ${result.confidence ? `
                    <div class="result-section">
                        <h4><i class="fas fa-chart-bar"></i> Confidence</h4>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${result.confidence * 100}%"></div>
                            <span class="confidence-text">${Math.round(result.confidence * 100)}%</span>
                        </div>
                    </div>
                ` : ''}
                
                ${result.recommendations ? `
                    <div class="result-section">
                        <h4><i class="fas fa-lightbulb"></i> Recommendations</h4>
                        <p>${result.recommendations}</p>
                    </div>
                ` : ''}
                
                <div class="result-timestamp">
                    <small><i class="fas fa-clock"></i> ${new Date().toLocaleString()}</small>
                </div>
            </div>
        `;
        
        this.analysisResults.innerHTML = html;
        this.showNotification('Analysis completed successfully', 'success');
    }

    // Upload functionality
    handleFileSelect(event) {
        const file = event.target.files[0];
        if (file) {
            this.processUploadedFile(file);
        }
    }

    handleDragOver(event) {
        event.preventDefault();
        event.currentTarget.classList.add('dragover');
    }

    handleDragLeave(event) {
        event.preventDefault();
        event.currentTarget.classList.remove('dragover');
    }

    handleDrop(event) {
        event.preventDefault();
        event.currentTarget.classList.remove('dragover');
        
        const files = event.dataTransfer.files;
        if (files.length > 0) {
            this.processUploadedFile(files[0]);
        }
    }

    processUploadedFile(file) {
        if (!file.type.startsWith('image/')) {
            this.showNotification('Please select a valid image file', 'error');
            return;
        }
        
        const reader = new FileReader();
        reader.onload = (e) => {
            const previewImage = document.getElementById('preview-image');
            const uploadPreview = document.getElementById('upload-preview');
            const uploadArea = document.getElementById('upload-area');
            
            previewImage.src = e.target.result;
            uploadPreview.style.display = 'block';
            uploadArea.style.display = 'none';
            
            // Store the base64 data for analysis
            this.uploadedImageData = e.target.result.split(',')[1];
        };
        reader.readAsDataURL(file);
    }

    async analyzeUploadedImage() {
        if (!this.uploadedImageData) {
            this.showNotification('No image to analyze', 'error');
            return;
        }
        
        try {
            this.showLoading(true);
            const result = await this.analyzeImage(this.uploadedImageData);
            
            const uploadResults = document.getElementById('upload-results');
            uploadResults.innerHTML = `
                <div class="upload-analysis">
                    <h3><i class="fas fa-search"></i> Analysis Results</h3>
                    ${this.formatAnalysisResults(result)}
                </div>
            `;
            
        } catch (error) {
            console.error('Error analyzing uploaded image:', error);
            this.showNotification('Analysis failed. Please try again.', 'error');
        } finally {
            this.showLoading(false);
        }
    }

    clearUpload() {
        const uploadPreview = document.getElementById('upload-preview');
        const uploadArea = document.getElementById('upload-area');
        const uploadResults = document.getElementById('upload-results');
        const fileInput = document.getElementById('file-input');
        
        uploadPreview.style.display = 'none';
        uploadArea.style.display = 'block';
        uploadResults.innerHTML = '';
        fileInput.value = '';
        this.uploadedImageData = null;
    }

    formatAnalysisResults(result) {
        if (result.error) {
            return `<div class="error-message"><p>Error: ${result.error}</p></div>`;
        }
        
        return `
            <div class="analysis-content">
                <p><strong>Description:</strong> ${result.description || 'No description available'}</p>
                ${result.findings ? `
                    <div class="findings">
                        <strong>Findings:</strong>
                        <ul>${result.findings.map(f => `<li>${f}</li>`).join('')}</ul>
                    </div>
                ` : ''}
                ${result.confidence ? `
                    <p><strong>Confidence:</strong> ${Math.round(result.confidence * 100)}%</p>
                ` : ''}
                ${result.recommendations ? `
                    <p><strong>Recommendations:</strong> ${result.recommendations}</p>
                ` : ''}
            </div>
        `;
    }

    launchARInterface() {
        if (this.checkARSupport()) {
            // In a real implementation, this would launch the AR interface
            this.showNotification('AR Interface would launch here (WebXR required)', 'info');
            
            // For demo purposes, show AR simulation
            this.showARSimulation();
        } else {
            this.showNotification('AR not supported on this device', 'error');
        }
    }

    showARSimulation() {
        // Create a simple AR simulation overlay
        const arOverlay = document.createElement('div');
        arOverlay.className = 'ar-simulation';
        arOverlay.innerHTML = `
            <div class="ar-content">
                <h2>AR Interface Simulation</h2>
                <p>This would be the AR view with medical image overlay</p>
                <div class="ar-controls">
                    <button class="btn btn-danger" onclick="this.parentElement.parentElement.parentElement.remove()">
                        <i class="fas fa-times"></i> Close AR
                    </button>
                </div>
            </div>
        `;
        
        arOverlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.8);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 2000;
        `;
        
        document.body.appendChild(arOverlay);
    }

    showLoading(show) {
        this.loadingOverlay.style.display = show ? 'flex' : 'none';
    }

    showNotification(message, type = 'info') {
        this.notification.textContent = message;
        this.notification.className = `notification ${type}`;
        this.notification.style.display = 'block';
        
        setTimeout(() => {
            this.notification.style.display = 'none';
        }, 5000);
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new ARAgent();
});

// Add some additional CSS for dynamic elements
const additionalStyles = `
    .confidence-bar {
        position: relative;
        background: #e5e7eb;
        height: 20px;
        border-radius: 10px;
        overflow: hidden;
    }
    
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #ef4444, #f59e0b, #34d399);
        transition: width 0.3s ease;
    }
    
    .confidence-text {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-weight: 600;
        color: white;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
    
    .result-section {
        margin-bottom: 20px;
        padding-bottom: 15px;
        border-bottom: 1px solid #e5e7eb;
    }
    
    .result-section:last-child {
        border-bottom: none;
    }
    
    .result-section h4 {
        margin-bottom: 10px;
        color: #1f2937;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .result-timestamp {
        text-align: right;
        margin-top: 15px;
        color: #6b7280;
    }
    
    .error-message {
        background: #fef2f2;
        border: 1px solid #fecaca;
        border-radius: 8px;
        padding: 15px;
        color: #dc2626;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .ar-simulation .ar-content {
        background: white;
        padding: 40px;
        border-radius: 12px;
        text-align: center;
        max-width: 500px;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
    }
`;

const styleSheet = document.createElement('style');
styleSheet.textContent = additionalStyles;
document.head.appendChild(styleSheet);