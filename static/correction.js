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
const elements = {};

document.addEventListener('DOMContentLoaded', () => {
    cacheElements();
    wireShotPreview(elements.topImage, elements.topPreview, elements.topStatus);
    wireShotPreview(elements.sideImage, elements.sidePreview, elements.sideStatus);
    wireShotPreview(elements.insideImage, elements.insidePreview, elements.insideStatus);
    wireShotPreview(elements.nutritionLabelImage, elements.nutritionLabelPreview, elements.nutritionLabelStatus, 'Optional image');
    applyDraftContext();

    elements.backToMainBtn.addEventListener('click', () => {
        window.location.href = 'index.html';
    });

    elements.correctionForm.addEventListener('submit', (event) => {
        event.preventDefault();
        submitCorrection();
    });
});

function cacheElements() {
    const ids = [
        'backToMainBtn', 'correctionForm', 'correctionLoading', 'correctionError', 'correctionSuccess',
        'correctionResult', 'mealName', 'foodType', 'brandName', 'portionSize', 'nutritionFacts',
        'proteinType', 'proteinAmount', 'addedItems', 'addedAmount', 'notes', 'predictedLabel',
        'predictionContext', 'draftContext', 'draftPreview', 'draftMealName', 'draftPredictionHint',
        'topImage', 'topPreview', 'topStatus', 'sideImage', 'sidePreview', 'sideStatus', 'insideImage',
        'insidePreview', 'insideStatus', 'nutritionLabelImage', 'nutritionLabelPreview', 'nutritionLabelStatus',
        'resultCalories', 'resultProtein', 'resultCarbs', 'resultFat', 'resultQuery'
    ];

    ids.forEach((id) => {
        elements[id] = document.getElementById(id);
    });
}

function applyDraftContext() {
    const raw = sessionStorage.getItem('nutrivisionCorrectionDraft');
    if (!raw) {
        return;
    }

    try {
        const draft = JSON.parse(raw);
        const initialMealName = draft.customFoodName || draft.selectedLabel || draft.predictedLabel || '';
        elements.mealName.value = formatFoodLabel(initialMealName);
        elements.foodType.value = formatFoodLabel(draft.predictedLabel || '');
        elements.predictedLabel.value = draft.predictedLabel || '';
        elements.notes.value = draft.mealComment || '';
        elements.predictionContext.value = JSON.stringify({
            predictions: draft.predictions || [],
            source: draft.source || null,
            datasetIndex: draft.datasetIndex ?? null
        });

        if (draft.previewUrl) {
            elements.draftPreview.src = draft.previewUrl;
            elements.draftContext.classList.remove('hidden');
        }

        elements.draftMealName.textContent = `Current prediction: ${formatFoodLabel(draft.predictedLabel || initialMealName || 'Unknown meal')}`;
        if (draft.customLookupStatus === 'not_found') {
            elements.draftPredictionHint.textContent = 'The quick custom meal search did not find a nutrition match. Upload 3 angles and fill in the meal name, type, brand, and nutrition facts so this sample can be queued for training review.';
        } else {
            elements.draftPredictionHint.textContent = 'Use this page if the result was wrong or you need more specific fields than the quick estimate screen.';
        }
        elements.draftContext.classList.remove('hidden');
    } catch (error) {
        sessionStorage.removeItem('nutrivisionCorrectionDraft');
    }
}

function wireShotPreview(input, preview, status, emptyLabel = 'Upload image') {
    input.addEventListener('change', () => {
        const [file] = input.files;
        if (!file) {
            preview.classList.add('hidden');
            preview.removeAttribute('src');
            status.textContent = emptyLabel;
            return;
        }

        const objectUrl = URL.createObjectURL(file);
        preview.src = objectUrl;
        preview.classList.remove('hidden');
        status.textContent = file.name;
    });
}

async function submitCorrection() {
    hideMessages();
    elements.correctionLoading.classList.remove('hidden');

    try {
        const formData = new FormData(elements.correctionForm);
        const response = await fetch(`${API_URL}/feedback/submit`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json().catch(() => ({}));
        if (!response.ok) {
            throw new Error(data.detail || data.message || `HTTP ${response.status}: ${response.statusText}`);
        }

        const submissionDir = data.entry?.submission_dir ? ` Saved in ${data.entry.submission_dir}.` : '';
        elements.correctionSuccess.textContent = `${data.message || 'Correction saved.'}${submissionDir}`;
        elements.correctionSuccess.classList.remove('hidden');
        renderNutritionResult(data.entry?.nutrition_result || null);
    } catch (error) {
        elements.correctionError.textContent = error.message || 'Failed to submit correction.';
        elements.correctionError.classList.remove('hidden');
    } finally {
        elements.correctionLoading.classList.add('hidden');
    }
}

function renderNutritionResult(result) {
    if (!result || result.status !== 'found' || !result.nutrition) {
        elements.resultQuery.innerHTML = '<strong>No calorie estimate found</strong><p>The correction was still saved for future review and training so the team can inspect it later.</p>';
        elements.correctionResult.classList.remove('hidden');
        return;
    }

    const nutrition = result.nutrition;
    elements.resultCalories.textContent = Math.round(nutrition.calories || 0);
    elements.resultProtein.textContent = `${Math.round(nutrition.protein_g || 0)}g`;
    elements.resultCarbs.textContent = `${Math.round(nutrition.carbs_g || 0)}g`;
    elements.resultFat.textContent = `${Math.round(nutrition.fat_g || 0)}g`;
    elements.resultQuery.innerHTML = `<strong>Query used</strong><p>${escapeHtml(result.query_used || 'Unknown')}</p>`;
    elements.correctionResult.classList.remove('hidden');
}

function hideMessages() {
    elements.correctionError.classList.add('hidden');
    elements.correctionError.textContent = '';
    elements.correctionSuccess.classList.add('hidden');
    elements.correctionSuccess.textContent = '';
}

function formatFoodLabel(label) {
    return String(label || '')
        .replace(/_/g, ' ')
        .replace(/\s+/g, ' ')
        .trim()
        .replace(/\b\w/g, (char) => char.toUpperCase());
}

function escapeHtml(text) {
    return String(text || '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
}
