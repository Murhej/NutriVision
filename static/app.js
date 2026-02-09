// API base URL
const API_URL = 'http://localhost:8000';

// Global state
let currentFile = null;
let currentDatasetIndex = null;
let currentSlide = 0;
let perClassData = null;
let currentSort = 'worst';

// Initialize app
document.addEventListener('DOMContentLoaded', async () => {
    await loadModelInfo();
    await loadCategories();
    setupEventListeners();
    initCarousel();
});

// Load model and training info
async function loadModelInfo() {
    try {
        const response = await fetch(`${API_URL}/info`);
        const data = await response.json();
        
        // Update header info
        const modelInfo = document.getElementById('modelInfo');
        modelInfo.innerHTML = `
            Model: <strong>${data.best_model_name}</strong> | 
            Top-3: <strong>${data.best_model_metrics.test_top3_accuracy.toFixed(2)}%</strong>
        `;
        
        // Update metrics cards
        document.getElementById('bestModel').textContent = data.best_model_name.toUpperCase();
        document.getElementById('top1Acc').textContent = `${data.best_model_metrics.test_top1_accuracy.toFixed(2)}%`;
        document.getElementById('top3Acc').textContent = `${data.best_model_metrics.test_top3_accuracy.toFixed(2)}%`;
        
    } catch (error) {
        console.error('Error loading model info:', error);
        document.getElementById('modelInfo').textContent = 'Error loading model info';
    }
}

// Setup event listeners
function setupEventListeners() {
    // Upload area
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    
    uploadArea.addEventListener('click', () => fileInput.click());
    
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });
    
    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });
    
    // Predict buttons
    document.getElementById('predictBtn').addEventListener('click', () => predictImage());
    document.getElementById('predictDatasetBtn').addEventListener('click', () => predictDatasetImage());
    
    // Random dataset button
    document.getElementById('randomBtn').addEventListener('click', () => loadRandomDatasetImage());
    
    // Category selector
    document.getElementById('categorySelect').addEventListener('change', (e) => {
        const btn = document.getElementById('categoryBtn');
        btn.disabled = !e.target.value;
    });
    
    document.getElementById('categoryBtn').addEventListener('click', () => loadCategoryImage());
    
    // Clear button
    document.getElementById('clearBtn').addEventListener('click', clearImage);
}

// Load food categories
async function loadCategories() {
    const select = document.getElementById('categorySelect');
    
    try {
        console.log('Loading categories from:', `${API_URL}/classes`);
        const response = await fetch(`${API_URL}/classes`);
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        console.log('Categories loaded:', data.classes.length);
        
        select.innerHTML = '<option value="">-- Select a food category --</option>';
        
        data.classes.forEach(className => {
            const option = document.createElement('option');
            option.value = className;
            option.textContent = className.replace(/_/g, ' ');
            select.appendChild(option);
        });
        
        console.log('✓ Category dropdown populated with', data.classes.length, 'items');
        
    } catch (error) {
        console.error('Error loading categories:', error);
        select.innerHTML = '<option value="">Error loading categories</option>';
    }
}

// Tab switching
function switchTab(tab) {
    // Update tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    event.target.classList.add('active');
    
    // Update tab content
    document.getElementById('uploadTab').classList.toggle('active', tab === 'upload');
    document.getElementById('datasetTab').classList.toggle('active', tab === 'dataset');
    
    // Clear previous results
    clearImage();
}

// Handle file upload
function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        showError('Please upload an image file');
        return;
    }
    
    currentFile = file;
    currentDatasetIndex = null;
    
    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        showImagePreview(e.target.result);
    };
    reader.readAsDataURL(file);
    
    // Enable predict button
    document.getElementById('predictBtn').disabled = false;
    
    // Hide results
    document.getElementById('results').style.display = 'none';
}

// Load random dataset image
async function loadRandomDatasetImage() {
    try {
        const response = await fetch(`${API_URL}/dataset/random`);
        const data = await response.json();
        
        currentDatasetIndex = data.index;
        currentFile = null;
        
        // Show dataset info
        const infoDiv = document.getElementById('datasetInfo');
        infoDiv.innerHTML = `
            <p><strong>Source:</strong> Random (Any Category)</p>
            <p><strong>Dataset Index:</strong> ${data.index}</p>
            <p><strong>True Label:</strong> ${data.true_label}</p>
        `;
        infoDiv.classList.add('active');
        
        // Load and show image
        const imageUrl = `${API_URL}/dataset/image/${data.index}`;
        showImagePreview(imageUrl);
        
        // Enable predict button
        document.getElementById('predictDatasetBtn').disabled = false;
        
        // Hide results
        document.getElementById('results').style.display = 'none';
        
    } catch (error) {
        showError('Error loading dataset image: ' + error.message);
    }
}

// Load image from specific category
async function loadCategoryImage() {
    const select = document.getElementById('categorySelect');
    const category = select.value;
    
    if (!category) return;
    
    try {
        const response = await fetch(`${API_URL}/dataset/random?category=${encodeURIComponent(category)}`);
        const data = await response.json();
        
        currentDatasetIndex = data.index;
        currentFile = null;
        
        // Show dataset info
        const infoDiv = document.getElementById('datasetInfo');
        infoDiv.innerHTML = `
            <p><strong>Source:</strong> Selected Category</p>
            <p><strong>True Label:</strong> ${data.true_label}</p>
            <p><strong>Dataset Index:</strong> ${data.index}</p>
        `;
        infoDiv.classList.add('active');
        
        // Load and show image
        const imageUrl = `${API_URL}/dataset/image/${data.index}`;
        showImagePreview(imageUrl);
        
        // Enable predict button
        document.getElementById('predictDatasetBtn').disabled = false;
        
        // Hide results
        document.getElementById('results').style.display = 'none';
        
    } catch (error) {
        showError('Error loading category image: ' + error.message);
    }
}

// Show image preview
function showImagePreview(src) {
    const preview = document.getElementById('imagePreview');
    const img = document.getElementById('previewImg');
    
    img.src = src;
    preview.style.display = 'block';
}

// Clear image
function clearImage() {
    currentFile = null;
    currentDatasetIndex = null;
    
    document.getElementById('imagePreview').style.display = 'none';
    document.getElementById('results').style.display = 'none';
    document.getElementById('predictBtn').disabled = true;
    document.getElementById('predictDatasetBtn').disabled = true;
    document.getElementById('datasetInfo').classList.remove('active');
    document.getElementById('fileInput').value = '';
}

// Predict uploaded image
async function predictImage() {
    if (!currentFile) return;
    
    showLoading(true);
    
    try {
        const formData = new FormData();
        formData.append('file', currentFile);
        
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            showResults(data.predictions);
        } else {
            showError('Prediction failed');
        }
        
    } catch (error) {
        showError('Error: ' + error.message);
    } finally {
        showLoading(false);
    }
}

// Predict dataset image
async function predictDatasetImage() {
    if (currentDatasetIndex === null) return;
    
    showLoading(true);
    
    try {
        // Get the image as blob
        const imageResponse = await fetch(`${API_URL}/dataset/image/${currentDatasetIndex}`);
        const blob = await imageResponse.blob();
        
        // Create file from blob
        const file = new File([blob], 'dataset_image.jpg', { type: 'image/jpeg' });
        
        // Predict
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Get true label from dataset info
            const infoDiv = document.getElementById('datasetInfo');
            const match = infoDiv.innerHTML.match(/True Label:<\/strong> (.+?)<\/p>/);
            const trueLabelText = match ? match[1] : null;
            
            showResults(data.predictions, trueLabelText);
        } else {
            showError('Prediction failed');
        }
        
    } catch (error) {
        showError('Error: ' + error.message);
    } finally {
        showLoading(false);
    }
}

// Show prediction results
function showResults(predictions, trueLabel = null) {
    const resultsDiv = document.getElementById('results');
    const listDiv = document.getElementById('predictionsList');
    const trueLabelDiv = document.getElementById('trueLabel');
    
    // Clear previous results
    listDiv.innerHTML = '';
    
    // Add prediction items
    predictions.forEach(pred => {
        const item = document.createElement('div');
        item.className = `prediction-item rank-${pred.rank}`;
        
        item.innerHTML = `
            <div class="rank-badge">${pred.rank}</div>
            <div class="prediction-details">
                <div class="prediction-class">${pred.class.replace(/_/g, ' ')}</div>
                <div class="prediction-confidence">
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${pred.confidence}%"></div>
                    </div>
                    <div class="confidence-text">${pred.confidence.toFixed(1)}%</div>
                </div>
            </div>
        `;
        
        listDiv.appendChild(item);
    });
    
    // Show true label if available (dataset images)
    if (trueLabel) {
        const topPrediction = predictions[0].class.replace(/_/g, ' ');
        const isCorrect = predictions.some(p => p.class === trueLabel.replace(/ /g, '_'));
        
        trueLabelDiv.innerHTML = `
            <strong>Ground Truth:</strong> ${trueLabel}<br>
            <strong>Result:</strong> ${isCorrect ? '✅ Correct (in top-3)' : '❌ Not in top-3'}
        `;
        trueLabelDiv.className = `true-label ${isCorrect ? 'correct' : 'incorrect'}`;
        trueLabelDiv.style.display = 'block';
    } else {
        trueLabelDiv.style.display = 'none';
    }
    
    resultsDiv.style.display = 'block';
}

// Show/hide loading
function showLoading(show) {
    document.getElementById('loading').style.display = show ? 'block' : 'none';
    document.getElementById('results').style.display = show ? 'none' : document.getElementById('results').style.display;
}

// Show error message
function showError(message) {
    const resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = `<div class="error">${message}</div>`;
    resultsDiv.style.display = 'block';
}

// Switch plots
function showPlot(modelName, plotType) {
    const plotImg = document.getElementById(plotType === 'loss' ? 'lossPlot' : 'accPlot');
    plotImg.src = `/plots/${modelName}_${plotType}`;
    
    // Update active button
    const buttons = event.target.parentElement.querySelectorAll('.model-btn');
    buttons.forEach(btn => btn.classList.remove('active'));
    event.target.classList.add('active');
}

// Switch confusion matrix
function showConfusion(modelName) {
    const plotImg = document.getElementById('confusionPlot');
    plotImg.src = `/plots/confusion_${modelName}`;
    
    // Update active button
    const buttons = event.target.parentElement.querySelectorAll('.model-btn');
    buttons.forEach(btn => btn.classList.remove('active'));
    event.target.classList.add('active');
}

// Click to enlarge plots
document.addEventListener('click', (e) => {
    if (e.target.classList.contains('plot-img')) {
        window.open(e.target.src, '_blank');
    }
});

// ===== CAROUSEL FUNCTIONALITY =====

function initCarousel() {
    const slides = document.querySelectorAll('.carousel-slide');
    const indicatorsContainer = document.getElementById('carouselIndicators');
    
    // Create indicators
    slides.forEach((_, index) => {
        const indicator = document.createElement('div');
        indicator.className = `carousel-indicator ${index === 0 ? 'active' : ''}`;
        indicator.onclick = () => goToSlide(index);
        indicatorsContainer.appendChild(indicator);
    });
    
    updateCarousel();
}

function carouselNav(direction) {
    const slides = document.querySelectorAll('.carousel-slide');
    currentSlide = (currentSlide + direction + slides.length) % slides.length;
    updateCarousel();
}

function goToSlide(index) {
    currentSlide = index;
    updateCarousel();
}

function updateCarousel() {
    const slidesContainer = document.getElementById('carouselSlides');
    const indicators = document.querySelectorAll('.carousel-indicator');
    
    slidesContainer.style.transform = `translateX(-${currentSlide * 100}%)`;
    
    indicators.forEach((indicator, index) => {
        indicator.classList.toggle('active', index === currentSlide);
    });
}

// ===== PER-CLASS PERFORMANCE =====

async function loadPerClassPerformance() {
    const loadingDiv = document.getElementById('performanceLoading');
    const gridDiv = document.getElementById('performanceGrid');
    const btn = document.getElementById('loadPerformanceBtn');
    const warning = document.getElementById('performanceWarning');
    
    btn.disabled = true;
    btn.textContent = 'Loading...';
    loadingDiv.style.display = 'block';
    gridDiv.innerHTML = '';
    
    try {
        const response = await fetch(`${API_URL}/analysis/per-class`);
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        
        console.log('API Response:', data); // Debug log
        
        if (!data.classes || !Array.isArray(data.classes)) {
            throw new Error('Invalid response format: classes array not found');
        }
        
        perClassData = data.classes;
        displayPerClassPerformance();
        
        btn.style.display = 'none';
        warning.classList.add('hidden');
        
    } catch (error) {
        console.error('Error details:', error);
        gridDiv.innerHTML = `<div class="error">Error loading per-class performance: ${error.message}</div>`;
        btn.disabled = false;
        btn.textContent = 'Retry Analysis';
    } finally {
        loadingDiv.style.display = 'none';
    }
}

function sortPerformance(sortType) {
    currentSort = sortType;
    
    // Update button states
    document.querySelectorAll('.sort-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    event.target.classList.add('active');
    
    if (perClassData) {
        displayPerClassPerformance();
    }
}

function displayPerClassPerformance() {
    const gridDiv = document.getElementById('performanceGrid');
    gridDiv.innerHTML = '';
    
    if (!perClassData || !Array.isArray(perClassData)) {
        console.error('perClassData is not valid:', perClassData);
        gridDiv.innerHTML = '<div class="error">No data available. Please load the analysis first.</div>';
        return;
    }
    
    // Sort data based on current sort type
    let sortedData = [...perClassData];
    
    if (currentSort === 'worst') {
        sortedData.sort((a, b) => a.accuracy - b.accuracy);
    } else if (currentSort === 'best') {
        sortedData.sort((a, b) => b.accuracy - a.accuracy);
    } else if (currentSort === 'name') {
        sortedData.sort((a, b) => a.class_name.localeCompare(b.class_name));
    }
    
    // Create cards
    sortedData.forEach(item => {
        const card = document.createElement('div');
        card.className = 'performance-card';
        
        // Determine quality class
        let qualityClass = 'poor';
        if (item.accuracy >= 80) qualityClass = 'excellent';
        else if (item.accuracy >= 60) qualityClass = 'good';
        else if (item.accuracy >= 40) qualityClass = 'fair';
        
        card.innerHTML = `
            <img 
                src="${API_URL}/dataset/image/${item.sample_image_index}" 
                alt="${item.class_name}"
                class="performance-image"
                onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%22200%22 height=%22150%22><rect width=%22200%22 height=%22150%22 fill=%22%23e0e0e0%22/><text x=%2250%%22 y=%2250%%22 text-anchor=%22middle%22 fill=%22%23999%22 font-size=%2214%22>No Image</text></svg>'"
            >
            <div class="performance-info">
                <div class="performance-name" title="${item.class_name.replace(/_/g, ' ')}">${item.class_name.replace(/_/g, ' ')}</div>
                <div class="performance-stats">${item.correct}/${item.total} correct</div>
                <div class="performance-bar-container">
                    <div class="performance-bar-fill ${qualityClass}" style="width: ${item.accuracy}%"></div>
                </div>
                <div class="performance-percentage">${item.accuracy}% accuracy</div>
            </div>
        `;
        
        // Add click to see sample prediction
        card.onclick = () => {
            window.open(`${API_URL}/dataset/image/${item.sample_image_index}`, '_blank');
        };
        
        gridDiv.appendChild(card);
    });
}
