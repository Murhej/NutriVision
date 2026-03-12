function resolveApiUrl() {
    if (!window.location.protocol.startsWith('http')) {
        return 'http://localhost:8000';
    }

    const isLocalHost = ['127.0.0.1', 'localhost'].includes(window.location.hostname);
    const isFastApiPort = window.location.port === '8000';

    if (isLocalHost && !isFastApiPort) {
        return 'http://localhost:8000';
    }

    return window.location.origin;
}

const API_URL = resolveApiUrl();
const DEFAULT_PORTIONS = {
    small: { id: 'small', label: 'Small', multiplier: 0.68, grams: 115, ounces: 4 },
    medium: { id: 'medium', label: 'Medium', multiplier: 1.0, grams: 170, ounces: 6 },
    large: { id: 'large', label: 'Large', multiplier: 1.32, grams: 225, ounces: 8 },
    extra_large: { id: 'extra_large', label: 'Extra Large', multiplier: 1.68, grams: 285, ounces: 10 }
};

const CUSTOM_LOOKUP_DEBOUNCE_MS = 450;

const state = {
    currentFile: null,
    currentDatasetIndex: null,
    currentDatasetSample: null,
    currentSource: null,
    previewUrl: '',
    previewTransferUrl: '',
    previewIsObjectUrl: false,
    predictions: [],
    selectedPredictionIndex: 0,
    nutritionByLabel: {},
    selectedEstimate: null,
    currentNutrition: null,
    portionPresets: DEFAULT_PORTIONS,
    portionUnits: [],
    selectedUnitId: 'serving',
    portionAmount: 1,
    customLookupStatus: 'idle',
    customLookupResult: null,
    customLookupRequestId: 0,
    customLookupTimer: null,
    customLookupContext: '',
    lastResolvedCustomName: ''
};

const elements = {};

document.addEventListener('DOMContentLoaded', async () => {
    cacheElements();
    setupEventListeners();
    setStep('idle');

    await Promise.allSettled([
        loadModelInfo(),
        loadCategories(),
        loadPortionPresets(),
        loadRecentLogs()
    ]);
});

function cacheElements() {
    const ids = [
        'backBtn', 'screenTitle', 'fileInput', 'randomBtn', 'categorySelect', 'categoryBtn',
        'modelInfo', 'datasetInfo', 'uploadArea', 'emptyState', 'imagePreview', 'previewImg',
        'heroMatchBadge', 'clearBtn', 'loading', 'loadingText', 'resultsError', 'saveMessage',
        'analysisStep', 'predictionOptions', 'customFoodInput', 'customLookupPanel', 'mealComment',
        'continueBtn', 'correctionLink', 'portionStep', 'summaryImg', 'portionMealName',
        'portionMealComment', 'kcalValue', 'proteinValue', 'carbsValue', 'fatValue',
        'portionAmountBadge', 'portionSlider', 'portionUnitSelect', 'portionHelper',
        'macroBarCarbs', 'macroBarFat', 'macroBarProtein', 'nutritionFactsList',
        'portionBackBtn', 'saveMealBtn', 'recentLogs', 'recentLogList'
    ];

    ids.forEach((id) => {
        elements[id] = document.getElementById(id);
    });
}

function setupEventListeners() {
    elements.fileInput.addEventListener('change', (event) => {
        const [file] = event.target.files;
        if (file) {
            handleFile(file);
        }
    });

    elements.uploadArea.addEventListener('click', (event) => {
        if (event.target === elements.clearBtn) {
            return;
        }
        elements.fileInput.click();
    });

    elements.uploadArea.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            elements.fileInput.click();
        }
    });

    elements.uploadArea.addEventListener('dragover', (event) => {
        event.preventDefault();
    });

    elements.uploadArea.addEventListener('drop', (event) => {
        event.preventDefault();
        const [file] = event.dataTransfer.files;
        if (file) {
            handleFile(file);
        }
    });

    elements.randomBtn.addEventListener('click', () => loadRandomDatasetImage());
    elements.categoryBtn.addEventListener('click', () => loadCategoryImage());
    elements.categorySelect.addEventListener('change', () => {
        elements.categoryBtn.disabled = !elements.categorySelect.value;
    });
    elements.clearBtn.addEventListener('click', (event) => {
        event.stopPropagation();
        clearImage();
    });
    elements.backBtn.addEventListener('click', () => handleBackAction());
    elements.continueBtn.addEventListener('click', () => continueToPortionStep());
    elements.portionBackBtn.addEventListener('click', () => setStep('analysis'));
    elements.saveMealBtn.addEventListener('click', () => saveMealLog());
    elements.correctionLink.addEventListener('click', () => openCorrectionPage());
    elements.portionSlider.addEventListener('input', () => {
        state.portionAmount = Number(elements.portionSlider.value);
        updatePortionSummary();
    });
    elements.portionUnitSelect.addEventListener('change', () => {
        state.selectedUnitId = elements.portionUnitSelect.value;
        configurePortionSlider();
        updatePortionSummary();
    });
    elements.customFoodInput.addEventListener('input', () => {
        handleCustomFoodInput();
        updateContinueButton();
    });
    elements.mealComment.addEventListener('input', () => {
        if (elements.customFoodInput.value.trim()) {
            scheduleCustomFoodLookup();
        }
        updateContinueButton();
    });
}

function buildApiUrl(path) {
    return `${API_URL}${path}`;
}

async function fetchJson(path, options = {}) {
    let response;
    try {
        response = await fetch(buildApiUrl(path), options);
    } catch (error) {
        throw new Error(`Cannot reach API at ${API_URL}. Start the FastAPI server and open http://localhost:8000/static/index.html`);
    }

    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
        throw new Error(data.detail || data.message || `HTTP ${response.status}: ${response.statusText}`);
    }
    return data;
}

async function postJson(path, body) {
    return fetchJson(path, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(body)
    });
}

async function loadModelInfo() {
    try {
        const data = await fetchJson('/info');
        elements.modelInfo.textContent = `Model ${String(data.best_model_name || '').toUpperCase()} | Top-3 ${(data.best_model_metrics?.test_top3_accuracy || 0).toFixed(2)}%`;
    } catch (error) {
        elements.modelInfo.textContent = 'Model info unavailable';
    }
}

async function loadCategories() {
    try {
        const data = await fetchJson('/classes');
        elements.categorySelect.innerHTML = '<option value="">Pick a dataset category</option>';
        data.classes.forEach((className) => {
            const option = document.createElement('option');
            option.value = className;
            option.textContent = formatFoodLabel(className);
            elements.categorySelect.appendChild(option);
        });
    } catch (error) {
        elements.categorySelect.innerHTML = '<option value="">Categories unavailable</option>';
    }
}

async function loadPortionPresets() {
    try {
        const data = await fetchJson('/map/portions');
        if (data.portions && typeof data.portions === 'object') {
            state.portionPresets = Object.fromEntries(
                Object.entries(data.portions).map(([key, value]) => [key, { ...value, id: key }])
            );
        }
    } catch (error) {
        state.portionPresets = DEFAULT_PORTIONS;
    }
}

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        showError('Please upload an image file.');
        return;
    }

    clearMessages();
    resetPredictionState();

    state.currentFile = file;
    state.currentDatasetIndex = null;
    state.currentDatasetSample = null;
    state.currentSource = 'upload';
    state.previewTransferUrl = '';

    showDatasetInfo('');
    showImagePreview(URL.createObjectURL(file), true);
    readFileAsDataUrl(file).then((result) => {
        state.previewTransferUrl = result;
    }).catch(() => {
        state.previewTransferUrl = '';
    });
    predictImage();
}

async function predictImage() {
    if (!state.currentFile) {
        return;
    }

    showLoading('Analyzing image...');
    try {
        const formData = new FormData();
        formData.append('file', state.currentFile);

        const response = await fetch(buildApiUrl('/predict'), {
            method: 'POST',
            body: formData
        });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) {
            throw new Error(data.detail || `HTTP ${response.status}: ${response.statusText}`);
        }

        handlePredictionResult(data);
    } catch (error) {
        showError(error.message || 'Prediction failed.');
    } finally {
        hideLoading();
    }
}

async function loadRandomDatasetImage() {
    clearMessages();
    resetPredictionState();
    showLoading('Loading dataset sample...');
    try {
        const data = await fetchJson('/dataset/random');
        state.currentSource = 'dataset';
        state.currentDatasetIndex = data.index;
        state.currentDatasetSample = data;
        state.previewTransferUrl = buildApiUrl(`/dataset/image/${data.index}`);
        showDatasetInfo(`Dataset sample | ${formatFoodLabel(data.true_label)} | #${data.index}`);
        showImagePreview(buildApiUrl(`/dataset/image/${data.index}`));
        await predictDatasetImage();
    } catch (error) {
        showError(`Could not load dataset image: ${error.message}`);
    } finally {
        hideLoading();
    }
}

async function loadCategoryImage() {
    const category = elements.categorySelect.value;
    if (!category) {
        return;
    }

    clearMessages();
    resetPredictionState();
    showLoading('Loading category sample...');
    try {
        const data = await fetchJson(`/dataset/random?category=${encodeURIComponent(category)}`);
        state.currentSource = 'dataset';
        state.currentDatasetIndex = data.index;
        state.currentDatasetSample = data;
        state.previewTransferUrl = buildApiUrl(`/dataset/image/${data.index}`);
        showDatasetInfo(`Dataset sample | ${formatFoodLabel(data.true_label)} | #${data.index}`);
        showImagePreview(buildApiUrl(`/dataset/image/${data.index}`));
        await predictDatasetImage();
    } catch (error) {
        showError(`Could not load category image: ${error.message}`);
    } finally {
        hideLoading();
    }
}

async function predictDatasetImage() {
    if (state.currentDatasetIndex === null) {
        return;
    }

    const data = await fetchJson(`/predict/dataset/${state.currentDatasetIndex}`);
    state.currentDatasetSample = data.sample || state.currentDatasetSample;
    handlePredictionResult(data);
}

function handlePredictionResult(data) {
    state.predictions = Array.isArray(data.predictions) ? data.predictions : [];
    state.selectedPredictionIndex = 0;
    state.selectedEstimate = null;
    state.currentNutrition = null;
    state.nutritionByLabel = {};

    resetCustomLookup();
    elements.customFoodInput.value = '';
    elements.mealComment.value = '';

    if (!state.predictions.length) {
        showError('The model did not return any predictions.');
        return;
    }

    const bestPrediction = state.predictions[0];
    elements.heroMatchBadge.textContent = `${bestPrediction.confidence.toFixed(0)}% match`;
    elements.heroMatchBadge.classList.remove('hidden');

    setStep('analysis');
    renderPredictionOptions();
    updateContinueButton();
    hydratePredictionNutrition();
}

async function hydratePredictionNutrition() {
    for (const prediction of state.predictions) {
        if (state.nutritionByLabel[prediction.class]) {
            continue;
        }

        try {
            const result = await postJson('/map/food', {
                food_label: prediction.class,
                portion_id: 'medium'
            });
            state.nutritionByLabel[prediction.class] = result;
        } catch (error) {
            state.nutritionByLabel[prediction.class] = {
                status: 'unknown_food',
                message: error.message
            };
        }

        renderPredictionOptions();
    }
}

function renderPredictionOptions() {
    elements.predictionOptions.innerHTML = '';
    const previewSrc = state.previewUrl || elements.previewImg.src;

    state.predictions.forEach((prediction, index) => {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = `prediction-option${index === state.selectedPredictionIndex ? ' is-selected' : ''}`;
        button.addEventListener('click', () => {
            state.selectedPredictionIndex = index;
            renderPredictionOptions();
        });

        const thumb = document.createElement('img');
        thumb.className = 'prediction-thumb';
        thumb.src = previewSrc;
        thumb.alt = formatFoodLabel(prediction.class);

        const textWrap = document.createElement('div');

        const title = document.createElement('button');
        title.type = 'button';
        title.className = 'prediction-title-link';
        title.textContent = truncateText(formatFoodLabel(prediction.class), 28);
        title.title = 'Prediction wrong? Open detailed correction';
        title.addEventListener('click', (event) => {
            event.stopPropagation();
            state.selectedPredictionIndex = index;
            renderPredictionOptions();
            openCorrectionPage(prediction.class);
        });

        const meta = document.createElement('div');
        meta.className = 'nutrition-meta';
        meta.textContent = predictionNutritionText(prediction.class);

        textWrap.appendChild(title);
        textWrap.appendChild(meta);

        const confidence = document.createElement('div');
        confidence.className = `confidence-pill ${confidenceClass(prediction.confidence)}`.trim();
        confidence.textContent = `${prediction.confidence.toFixed(0)}%`;

        button.appendChild(thumb);
        button.appendChild(textWrap);
        button.appendChild(confidence);
        elements.predictionOptions.appendChild(button);
    });
}

function predictionNutritionText(label) {
    const mapped = state.nutritionByLabel[label];
    if (!mapped) {
        return 'Estimating nutrition...';
    }

    if (mapped.status === 'found' && mapped.nutrition) {
        return `${Math.round(mapped.nutrition.calories || 0)} kcal | ${Math.round(mapped.nutrition.protein_g || 0)}g protein`;
    }

    return 'Nutrition estimate unavailable';
}

function confidenceClass(confidence) {
    if (confidence >= 85) {
        return 'high';
    }
    if (confidence >= 70) {
        return 'medium';
    }
    return '';
}

function handleCustomFoodInput() {
    const customName = elements.customFoodInput.value.trim();
    clearMessages();

    if (!customName) {
        resetCustomLookup();
        renderCustomLookupPanel();
        return;
    }

    scheduleCustomFoodLookup();
}

function scheduleCustomFoodLookup() {
    const customName = elements.customFoodInput.value.trim();
    if (!customName) {
        resetCustomLookup();
        renderCustomLookupPanel();
        return;
    }

    if (customName.length < 3) {
        state.customLookupStatus = 'typing';
        state.customLookupResult = null;
        renderCustomLookupPanel();
        return;
    }

    window.clearTimeout(state.customLookupTimer);
    state.customLookupStatus = 'searching';
    state.customLookupResult = null;
    renderCustomLookupPanel();

    state.customLookupTimer = window.setTimeout(() => {
        lookupCustomFood(customName, elements.mealComment.value.trim());
    }, CUSTOM_LOOKUP_DEBOUNCE_MS);
}

async function lookupCustomFood(customName, comment) {
    const requestId = ++state.customLookupRequestId;
    state.customLookupContext = comment;

    try {
        const result = await postJson('/map/food', {
            food_label: customName,
            user_description: comment || null,
            portion_id: 'medium'
        });

        if (requestId !== state.customLookupRequestId) {
            return;
        }

        state.customLookupResult = result;
        state.lastResolvedCustomName = customName;
        state.customLookupStatus = result.status === 'found' ? 'found' : 'not_found';
    } catch (error) {
        if (requestId !== state.customLookupRequestId) {
            return;
        }

        state.customLookupStatus = 'error';
        state.customLookupResult = { message: error.message || 'Lookup failed.' };
    }

    renderCustomLookupPanel();
    updateContinueButton();
}

function renderCustomLookupPanel() {
    const customName = elements.customFoodInput.value.trim();
    const status = state.customLookupStatus;

    if (!customName) {
        elements.customLookupPanel.classList.add('hidden');
        elements.customLookupPanel.innerHTML = '';
        return;
    }

    elements.customLookupPanel.classList.remove('hidden');

    if (status === 'typing') {
        elements.customLookupPanel.innerHTML = `
            <div class="lookup-card lookup-card-searching">
                <div class="lookup-title">Keep typing</div>
                <p class="lookup-meta">Enter at least 3 letters to search the nutrition API with your custom meal name.</p>
            </div>
        `;
        return;
    }

    if (status === 'searching') {
        elements.customLookupPanel.innerHTML = `
            <div class="lookup-card lookup-card-searching">
                <div class="spinner"></div>
                <div>
                    <div class="lookup-title">Searching for "${escapeHtml(customName)}"</div>
                    <p class="lookup-meta">Trying your meal name and notes against the food API.</p>
                </div>
            </div>
        `;
        return;
    }

    if (status === 'found' && state.customLookupResult?.nutrition) {
        const nutrition = state.customLookupResult.nutrition;
        elements.customLookupPanel.innerHTML = `
            <div class="lookup-card lookup-card-found">
                <div class="lookup-title">Custom meal found</div>
                <div class="lookup-grid">
                    <div><strong>${Math.round(nutrition.calories || 0)}</strong><span>kcal</span></div>
                    <div><strong>${Math.round(nutrition.protein_g || 0)}g</strong><span>protein</span></div>
                    <div><strong>${Math.round(nutrition.carbs_g || 0)}g</strong><span>carbs</span></div>
                    <div><strong>${Math.round(nutrition.fat_g || 0)}g</strong><span>fat</span></div>
                </div>
                <p class="lookup-meta">Using query: ${escapeHtml(state.customLookupResult.query_used || customName)}</p>
                <p class="lookup-hint">This custom name will override the model prediction when you continue.</p>
            </div>
        `;
        return;
    }

    if (status === 'not_found') {
        const followUpQuestions = renderLookupQuestionList(state.customLookupResult?.follow_up_questions);
        const externalApiQueries = renderLookupQueryList(state.customLookupResult?.external_api_queries);
        elements.customLookupPanel.innerHTML = `
            <div class="lookup-card lookup-card-missing">
                <img class="lookup-illustration" src="${buildNotFoundIllustration(customName)}" alt="Food not found">
                <div>
                    <div class="lookup-title">Food not found</div>
                    <p class="lookup-meta">"${escapeHtml(customName)}" did not return nutrition data. Use the correction page to upload 3 angles, add the real meal name, type, brand, and nutrition facts for training review.</p>
                    ${followUpQuestions ? `<div class="lookup-meta"><strong>Answer these next:</strong>${followUpQuestions}</div>` : ''}
                    ${externalApiQueries ? `<div class="lookup-meta"><strong>Better API search ideas:</strong>${externalApiQueries}</div>` : ''}
                    <button type="button" class="inline-link-button lookup-action-button" data-action="correction">Open correction page</button>
                </div>
            </div>
        `;
        const actionButton = elements.customLookupPanel.querySelector('[data-action="correction"]');
        if (actionButton) {
            actionButton.addEventListener('click', () => openCorrectionPage(customName));
        }
        return;
    }

    elements.customLookupPanel.innerHTML = `
        <div class="lookup-card lookup-card-missing">
            <div class="lookup-title">Lookup failed</div>
            <p class="lookup-meta">${escapeHtml(state.customLookupResult?.message || 'Could not search the nutrition API right now.')}</p>
        </div>
    `;
}

function updateContinueButton() {
    const hasPrediction = state.predictions.length > 0;
    const customName = elements.customFoodInput.value.trim();

    if (!hasPrediction) {
        elements.continueBtn.disabled = true;
        return;
    }

    if (!customName) {
        elements.continueBtn.disabled = false;
        return;
    }

    elements.continueBtn.disabled = state.customLookupStatus !== 'found';
}

async function continueToPortionStep() {
    const selectedLabel = getSelectedFoodLabel();
    if (!selectedLabel) {
        showError('Choose a prediction or enter a custom meal name.');
        return;
    }

    if (elements.customFoodInput.value.trim() && state.customLookupStatus !== 'found') {
        showError('Your custom meal name needs a valid nutrition match. If it is missing, open the correction page.');
        return;
    }

    clearMessages();
    showLoading('Estimating nutrition...');

    try {
        const result = await postJson('/map/food', {
            food_label: selectedLabel,
            user_description: elements.mealComment.value.trim() || null,
            portion_id: 'medium'
        });

        if (result.status === 'provider_error') {
            const triedQuery = result.query_used || result.external_api_queries?.[0] || selectedLabel;
            const providerName = String(result.provider || 'nutrition provider').replaceAll('_', ' ');
            throw new Error(`${result.message || `Could not reach ${providerName}.`} Last query: ${triedQuery}`);
        }

        if (result.status !== 'found') {
            const nextBestQuery = result.next_best_query || result.external_api_queries?.[0] || selectedLabel;
            const followUp = Array.isArray(result.follow_up_questions) && result.follow_up_questions.length
                ? ` Add details like: ${result.follow_up_questions.slice(0, 2).join(' ')}`
                : '';
            throw new Error(`Edamam could not match "${formatFoodLabel(selectedLabel)}" yet. Tried query: ${nextBestQuery}.${followUp}`);
        }

        state.selectedEstimate = result;
        buildPortionUnits(selectedLabel, result);
        setStep('portion');
        updatePortionSummary();
    } catch (error) {
        showError(error.message || 'Nutrition estimate failed.');
    } finally {
        hideLoading();
    }
}

function buildPortionUnits(label, result) {
    const normalized = String(label || '').toLowerCase();
    const mediumPortion = state.portionPresets.medium || DEFAULT_PORTIONS.medium;
    const baseGrams = Number(result?.portion?.grams || mediumPortion.grams || 170);
    const baseOunces = Number(result?.portion?.ounces || mediumPortion.ounces || 6);
    const units = [
        {
            id: 'serving',
            label: 'Serving',
            factor: 1,
            min: 0.5,
            max: 3,
            step: 0.25,
            formatter: (amount) => `${formatAmount(amount)} serving${amount === 1 ? '' : 's'}`
        },
        {
            id: 'grams_100',
            label: '100 g',
            factor: 100 / baseGrams,
            min: 0.5,
            max: 4,
            step: 0.5,
            formatter: (amount) => `${formatAmount(amount * 100)} g`
        },
        {
            id: 'oz',
            label: 'Oz',
            factor: 1 / baseOunces,
            min: 1,
            max: 16,
            step: 1,
            formatter: (amount) => `${formatAmount(amount)} oz`
        }
    ];

    if (normalized.includes('pizza')) {
        units.unshift({
            id: 'slice',
            label: 'Slice',
            factor: 0.5,
            min: 1,
            max: 6,
            step: 1,
            formatter: (amount) => `${formatAmount(amount)} slice${amount === 1 ? '' : 's'}`
        });
    } else if (/(sushi|dumpling|wing|nugget|cookie|piece|roll)/.test(normalized)) {
        units.unshift({
            id: 'piece',
            label: 'Piece',
            factor: 0.25,
            min: 2,
            max: 12,
            step: 1,
            formatter: (amount) => `${formatAmount(amount)} pieces`
        });
    } else if (/(pasta|salad|rice|soup|cereal|noodle)/.test(normalized)) {
        units.unshift({
            id: 'cup',
            label: 'Cup',
            factor: 0.67,
            min: 0.5,
            max: 4,
            step: 0.25,
            formatter: (amount) => `${formatAmount(amount)} cup${amount === 1 ? '' : 's'}`
        });
    } else if (/(burger|sandwich|taco|burrito|wrap)/.test(normalized)) {
        units.unshift({
            id: 'item',
            label: 'Item',
            factor: 1,
            min: 0.5,
            max: 3,
            step: 0.5,
            formatter: (amount) => `${formatAmount(amount)} item${amount === 1 ? '' : 's'}`
        });
    }

    state.portionUnits = units;
    state.selectedUnitId = units[0]?.id || 'serving';
    state.portionAmount = units[0]?.min || 1;
    renderPortionUnitOptions();
    configurePortionSlider();
}

function renderPortionUnitOptions() {
    elements.portionUnitSelect.innerHTML = '';
    state.portionUnits.forEach((unit) => {
        const option = document.createElement('option');
        option.value = unit.id;
        option.textContent = unit.label;
        elements.portionUnitSelect.appendChild(option);
    });
    elements.portionUnitSelect.value = state.selectedUnitId;
}

function configurePortionSlider() {
    const unit = getActivePortionUnit();
    if (!unit) {
        return;
    }

    elements.portionSlider.min = String(unit.min);
    elements.portionSlider.max = String(unit.max);
    elements.portionSlider.step = String(unit.step);
    state.portionAmount = clamp(state.portionAmount || unit.min, unit.min, unit.max);
    state.portionAmount = snapToStep(state.portionAmount, unit.step, unit.min);
    elements.portionSlider.value = String(state.portionAmount);
    updateSliderScale(unit);
}

function updateSliderScale(unit) {
    const scaleMarks = document.querySelectorAll('.slider-scale span');
    if (!scaleMarks.length) {
        return;
    }

    const values = [
        unit.min,
        unit.min + ((unit.max - unit.min) / 3),
        unit.min + ((unit.max - unit.min) * 2 / 3),
        unit.max
    ].map((value) => snapToStep(value, unit.step, unit.min));

    scaleMarks.forEach((mark, index) => {
        const value = values[index] ?? unit.max;
        mark.textContent = formatAmount(value);
    });
}

function updatePortionSummary() {
    if (!state.selectedEstimate) {
        return;
    }

    const unit = getActivePortionUnit();
    if (!unit) {
        return;
    }

    const multiplier = Number((state.portionAmount * unit.factor).toFixed(3));
    const baseNutrition = state.selectedEstimate.base_nutrition || state.selectedEstimate.nutrition || {};
    const scaledNutrition = scaleNutrition(baseNutrition, multiplier);

    state.currentNutrition = scaledNutrition;
    elements.summaryImg.src = state.previewUrl || elements.previewImg.src;
    elements.portionMealName.textContent = formatFoodLabel(getSelectedFoodLabel());
    elements.portionMealComment.textContent = elements.mealComment.value.trim()
        || `Estimated from: ${state.selectedEstimate.query_used}`;

    elements.portionAmountBadge.textContent = unit.formatter(state.portionAmount);
    elements.portionHelper.textContent = `${unit.formatter(state.portionAmount)} is about x${multiplier.toFixed(2)} of the API serving size.`;

    elements.kcalValue.textContent = Math.round(scaledNutrition.calories || 0);
    elements.proteinValue.textContent = `${Math.round(scaledNutrition.protein_g || 0)}g`;
    elements.carbsValue.textContent = `${Math.round(scaledNutrition.carbs_g || 0)}g`;
    elements.fatValue.textContent = `${Math.round(scaledNutrition.fat_g || 0)}g`;

    renderMacroBar(scaledNutrition);
    renderNutritionFacts(scaledNutrition);
}

function renderMacroBar(nutrition) {
    const proteinCalories = Math.max(0, Number(nutrition.protein_g || 0) * 4);
    const carbsCalories = Math.max(0, Number(nutrition.carbs_g || 0) * 4);
    const fatCalories = Math.max(0, Number(nutrition.fat_g || 0) * 9);
    const total = proteinCalories + carbsCalories + fatCalories;

    const carbsWidth = total ? (carbsCalories / total) * 100 : 0;
    const fatWidth = total ? (fatCalories / total) * 100 : 0;
    const proteinWidth = total ? (proteinCalories / total) * 100 : 0;

    elements.macroBarCarbs.style.width = `${carbsWidth}%`;
    elements.macroBarFat.style.width = `${fatWidth}%`;
    elements.macroBarProtein.style.width = `${proteinWidth}%`;
}

function renderNutritionFacts(nutrition) {
    const facts = [
        ['Calories', Math.round(nutrition.calories || 0), 'kcal'],
        ['Protein', Math.round(nutrition.protein_g || 0), 'g'],
        ['Carbs', Math.round(nutrition.carbs_g || 0), 'g'],
        ['Fat', Math.round(nutrition.fat_g || 0), 'g'],
        ['Fiber', Math.round(nutrition.fiber_g || 0), 'g'],
        ['Sugar', Math.round(nutrition.sugar_g || 0), 'g'],
        ['Sodium', Math.round(nutrition.sodium_mg || 0), 'mg'],
        ['Cholesterol', Math.round(nutrition.cholesterol_mg || 0), 'mg']
    ];

    elements.nutritionFactsList.innerHTML = '';
    facts.forEach(([label, value, unit]) => {
        const item = document.createElement('div');
        item.className = 'nutrition-fact-item';
        item.innerHTML = `<span>${label}</span><strong>${value}${unit}</strong>`;
        elements.nutritionFactsList.appendChild(item);
    });
}

async function saveMealLog() {
    if (!state.selectedEstimate || !state.currentNutrition) {
        showError('Estimate the meal first.');
        return;
    }

    const unit = getActivePortionUnit();
    const portionMultiplier = Number((state.portionAmount * (unit?.factor || 1)).toFixed(3));
    const selectedPrediction = state.predictions[state.selectedPredictionIndex] || null;

    clearMessages();
    showLoading('Saving meal log...');

    try {
        const data = await postJson('/map/log', {
            food_label: getSelectedFoodLabel(),
            display_name: formatFoodLabel(getSelectedFoodLabel()),
            comment: elements.mealComment.value.trim() || null,
            portion_id: unit?.id || 'serving',
            portion_label: unit ? unit.formatter(state.portionAmount) : '1 serving',
            portion_multiplier: portionMultiplier,
            nutrition: state.currentNutrition,
            prediction: selectedPrediction,
            source: state.currentSource,
            image_url: state.currentDatasetIndex !== null ? buildApiUrl(`/dataset/image/${state.currentDatasetIndex}`) : null
        });

        showSaveMessage(`Saved ${data.entry.display_name} to the meal log.`);
        await loadRecentLogs();
    } catch (error) {
        showError(error.message || 'Could not save meal log.');
    } finally {
        hideLoading();
    }
}

async function loadRecentLogs() {
    try {
        const data = await fetchJson('/map/logs?limit=3');
        renderRecentLogs(data.entries || []);
    } catch (error) {
        renderRecentLogs([]);
    }
}

function renderRecentLogs(entries) {
    elements.recentLogList.innerHTML = '';
    if (!entries.length) {
        elements.recentLogs.classList.add('hidden');
        return;
    }

    entries.forEach((entry) => {
        const item = document.createElement('article');
        item.className = 'log-item';

        const title = document.createElement('div');
        title.className = 'log-item-title';
        title.textContent = entry.display_name || formatFoodLabel(entry.food_label);

        const meta = document.createElement('div');
        meta.className = 'log-meta';
        meta.textContent = `${entry.portion_label || 'Portion'}${entry.comment ? ` | ${truncateText(entry.comment, 64)}` : ''}`;

        const nutrition = document.createElement('div');
        nutrition.className = 'log-item-nutrition';
        nutrition.textContent = `${Math.round(entry.nutrition?.calories || 0)} kcal | ${Math.round(entry.nutrition?.protein_g || 0)}g protein | ${Math.round(entry.nutrition?.carbs_g || 0)}g carbs`;

        item.appendChild(title);
        item.appendChild(meta);
        item.appendChild(nutrition);
        elements.recentLogList.appendChild(item);
    });

    elements.recentLogs.classList.remove('hidden');
}

function showImagePreview(src, isObjectUrl = false) {
    revokePreviewUrl();
    state.previewUrl = src;
    state.previewIsObjectUrl = isObjectUrl;
    if (!isObjectUrl) {
        state.previewTransferUrl = src;
    }

    elements.previewImg.src = src;
    elements.imagePreview.classList.remove('hidden');
    elements.emptyState.classList.add('hidden');
}

function revokePreviewUrl() {
    if (state.previewIsObjectUrl && state.previewUrl) {
        URL.revokeObjectURL(state.previewUrl);
    }
}

function clearImage() {
    revokePreviewUrl();
    window.clearTimeout(state.customLookupTimer);

    state.currentFile = null;
    state.currentDatasetIndex = null;
    state.currentDatasetSample = null;
    state.currentSource = null;
    state.previewUrl = '';
    state.previewTransferUrl = '';
    state.previewIsObjectUrl = false;

    elements.fileInput.value = '';
    elements.previewImg.removeAttribute('src');
    elements.summaryImg.removeAttribute('src');
    elements.imagePreview.classList.add('hidden');
    elements.emptyState.classList.remove('hidden');
    elements.heroMatchBadge.classList.add('hidden');
    elements.customFoodInput.value = '';
    elements.mealComment.value = '';
    elements.portionUnitSelect.innerHTML = '';
    elements.portionAmountBadge.textContent = '1.00 serving';
    elements.portionHelper.textContent = 'Select unit and intake amount.';
    elements.nutritionFactsList.innerHTML = '';
    elements.macroBarCarbs.style.width = '0%';
    elements.macroBarFat.style.width = '0%';
    elements.macroBarProtein.style.width = '0%';
    showDatasetInfo('');
    resetPredictionState();
    clearMessages();
    setStep('idle');
}

function resetPredictionState() {
    state.predictions = [];
    state.selectedPredictionIndex = 0;
    state.nutritionByLabel = {};
    state.selectedEstimate = null;
    state.currentNutrition = null;
    state.portionUnits = [];
    state.selectedUnitId = 'serving';
    state.portionAmount = 1;

    resetCustomLookup();
    elements.predictionOptions.innerHTML = '';
    renderCustomLookupPanel();
    updateContinueButton();
}

function resetCustomLookup() {
    window.clearTimeout(state.customLookupTimer);
    state.customLookupRequestId += 1;
    state.customLookupStatus = 'idle';
    state.customLookupResult = null;
    state.customLookupContext = '';
    state.lastResolvedCustomName = '';
}

function handleBackAction() {
    if (!elements.portionStep.classList.contains('hidden')) {
        setStep('analysis');
        return;
    }

    if (!elements.analysisStep.classList.contains('hidden') || state.previewUrl) {
        clearImage();
    }
}

function setStep(step) {
    elements.analysisStep.classList.toggle('hidden', step !== 'analysis');
    elements.portionStep.classList.toggle('hidden', step !== 'portion');
    elements.screenTitle.textContent = step === 'portion' ? 'Portion Size' : 'Food Analysis';
}

function showDatasetInfo(text) {
    if (!text) {
        elements.datasetInfo.textContent = '';
        elements.datasetInfo.classList.add('hidden');
        return;
    }

    elements.datasetInfo.textContent = text;
    elements.datasetInfo.classList.remove('hidden');
}

function showLoading(message) {
    elements.loadingText.textContent = message;
    elements.loading.classList.remove('hidden');
}

function hideLoading() {
    elements.loading.classList.add('hidden');
}

function clearMessages() {
    elements.resultsError.classList.add('hidden');
    elements.resultsError.textContent = '';
    elements.saveMessage.classList.add('hidden');
    elements.saveMessage.textContent = '';
}

function showError(message) {
    elements.resultsError.textContent = message;
    elements.resultsError.classList.remove('hidden');
    elements.saveMessage.classList.add('hidden');
}

function showSaveMessage(message) {
    elements.saveMessage.textContent = message;
    elements.saveMessage.classList.remove('hidden');
    elements.resultsError.classList.add('hidden');
}

function getSelectedFoodLabel() {
    const customName = elements.customFoodInput.value.trim();
    if (customName) {
        return customName;
    }
    return state.predictions[state.selectedPredictionIndex]?.class || '';
}

function getActivePortionUnit() {
    return state.portionUnits.find((unit) => unit.id === state.selectedUnitId) || state.portionUnits[0] || null;
}

function openCorrectionPage(forcedLabel = '') {
    const selectedLabel = forcedLabel || getSelectedFoodLabel();
    const selectedPrediction = state.predictions.find((prediction) => prediction.class === forcedLabel)
        || state.predictions[state.selectedPredictionIndex]
        || null;

    const payload = {
        selectedLabel,
        predictedLabel: selectedPrediction?.class || '',
        predictions: state.predictions,
        mealComment: elements.mealComment.value.trim(),
        customFoodName: elements.customFoodInput.value.trim(),
        customLookupStatus: state.customLookupStatus,
        customLookupQuery: state.customLookupResult?.query_used || '',
        customLookupMessage: state.customLookupResult?.message || '',
        previewUrl: state.previewTransferUrl || '',
        source: state.currentSource,
        datasetIndex: state.currentDatasetIndex
    };

    sessionStorage.setItem('nutrivisionCorrectionDraft', JSON.stringify(payload));
    window.location.href = 'correction.html';
}

function readFileAsDataUrl(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = () => reject(reader.error);
        reader.readAsDataURL(file);
    });
}

function scaleNutrition(nutrition, multiplier) {
    const scaled = {};
    Object.entries(nutrition || {}).forEach(([key, value]) => {
        scaled[key] = typeof value === 'number' ? Number((value * multiplier).toFixed(1)) : value;
    });
    return scaled;
}

function formatFoodLabel(label) {
    return String(label || '')
        .replace(/_/g, ' ')
        .replace(/\s+/g, ' ')
        .trim()
        .replace(/\b\w/g, (char) => char.toUpperCase());
}

function formatAmount(value) {
    return Number(value).toFixed(Number.isInteger(Number(value)) ? 0 : 2).replace(/\.00$/, '').replace(/(\.\d*[1-9])0$/, '$1');
}

function truncateText(text, maxLength) {
    const normalized = String(text || '');
    if (normalized.length <= maxLength) {
        return normalized;
    }
    return `${normalized.slice(0, maxLength - 1)}...`;
}

function escapeHtml(text) {
    return String(text || '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
}

function renderLookupQuestionList(items) {
    if (!Array.isArray(items) || !items.length) {
        return '';
    }
    return `<ul class="lookup-inline-list">${items.map((item) => `<li>${escapeHtml(item)}</li>`).join('')}</ul>`;
}

function renderLookupQueryList(items) {
    if (!Array.isArray(items) || !items.length) {
        return '';
    }
    return `<ul class="lookup-inline-list">${items.slice(0, 4).map((item) => `<li><code>${escapeHtml(item)}</code></li>`).join('')}</ul>`;
}

function clamp(value, min, max) {
    return Math.min(Math.max(Number(value), Number(min)), Number(max));
}

function snapToStep(value, step, min) {
    const offset = Number(value) - Number(min);
    const snapped = Math.round(offset / Number(step)) * Number(step) + Number(min);
    return Number(snapped.toFixed(2));
}

function buildNotFoundIllustration(label) {
    const safeLabel = escapeHtml(truncateText(label, 18));
    const svg = `
        <svg xmlns="http://www.w3.org/2000/svg" width="200" height="140" viewBox="0 0 200 140">
            <defs>
                <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stop-color="#0b2018"/>
                    <stop offset="100%" stop-color="#04100b"/>
                </linearGradient>
            </defs>
            <rect width="200" height="140" rx="24" fill="url(#bg)"/>
            <circle cx="100" cy="58" r="28" fill="none" stroke="#16d592" stroke-width="4" stroke-dasharray="6 6"/>
            <path d="M100 42c-8 0-14 5-14 13h8c0-4 3-6 6-6s6 2 6 6c0 3-2 5-4 7-5 4-8 8-8 15h8c0-4 1-6 5-9 4-3 7-7 7-13 0-8-6-13-14-13z" fill="#f4f7f2"/>
            <circle cx="100" cy="84" r="4" fill="#f4f7f2"/>
            <text x="100" y="116" font-family="Segoe UI, sans-serif" font-size="13" fill="#91a095" text-anchor="middle">${safeLabel}</text>
        </svg>
    `.trim();
    return `data:image/svg+xml;charset=UTF-8,${encodeURIComponent(svg)}`;
}
