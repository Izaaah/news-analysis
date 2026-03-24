// Set current date
function setCurrentDate() {
    const now = new Date();
    const options = { 
        weekday: 'long', 
        year: 'numeric', 
        month: 'long', 
        day: 'numeric' 
    };
    const dateStr = now.toLocaleDateString('id-ID', options);
    document.getElementById('current-date').textContent = dateStr;
}

// Update character count
function updateCharCount() {
    const input = document.getElementById('news-input');
    const charCount = document.getElementById('char-count');
    charCount.textContent = input.value.length;
}

// Format category name
function formatCategory(category) {
    const categoryMap = {
        'ekonomi': 'Ekonomi',
        'teknologi': 'Teknologi',
        'lifestyle': 'Gaya Hidup'
    };
    return categoryMap[category.toLowerCase()] || category;
}

// Get category icon
function getCategoryIcon(category) {
    const iconMap = {
        'ekonomi': 'fa-chart-line',
        'teknologi': 'fa-microchip',
        'lifestyle': 'fa-heart'
    };
    return iconMap[category.toLowerCase()] || 'fa-folder-open';
}

// Display results
function displayResults(data) {
    const resultsSection = document.getElementById('results-section');
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

    // Set category
    const categoryName = formatCategory(data.predicted_category);
    const categoryIcon = getCategoryIcon(data.predicted_category);
    const categoryBadge = document.getElementById('category-badge-large');
    
    categoryBadge.innerHTML = `
        <i class="fas ${categoryIcon}"></i>
        <span id="category-name">${categoryName}</span>
    `;

    // Set confidence
    const confidence = (data.confidence * 100).toFixed(1);
    document.getElementById('confidence-value').textContent = `${confidence}%`;
    setTimeout(() => {
        document.getElementById('confidence-fill').style.width = `${confidence}%`;
    }, 100);

    // Set original text
    document.getElementById('original-text').textContent = data.input_text;

    // Set article date
    const now = new Date();
    const dateStr = now.toLocaleDateString('id-ID', { 
        day: 'numeric', 
        month: 'long', 
        year: 'numeric' 
    });
    document.getElementById('article-date').textContent = dateStr;

    // Display similar news
    displaySimilarNews(data.similar_news);
    
    // Display summary
    console.log("Summary data:", data.summary, "Type:", typeof data.summary); // Debug log
    const summaryCard = document.getElementById('summary-card');
    const summaryText = document.getElementById('summary-text');
    
    if (data.summary && typeof data.summary === 'string' && data.summary.trim() && data.summary.trim() !== '') {
        summaryText.textContent = data.summary.trim();
        summaryCard.style.display = 'block';
        console.log("✓ Summary card displayed");
    } else {
        summaryCard.style.display = 'none';
        console.log("✗ Summary card hidden - no valid summary. Value:", data.summary);
    }

    // Display keywords
    console.log("Keywords data:", data.keywords, "Type:", typeof data.keywords, "Is Array:", Array.isArray(data.keywords)); // Debug log
    const keywordCard = document.getElementById('keyword-card');
    const keywordContainer = document.getElementById('keyword-container');
    
    if (data.keywords && Array.isArray(data.keywords) && data.keywords.length > 0) {
        keywordContainer.innerHTML = '';

        let validKeywords = 0;
        data.keywords.forEach((kw, idx) => {
            console.log(`Keyword ${idx}:`, kw, "Type:", typeof kw);
            if (kw && (typeof kw === 'string' || typeof kw === 'number') && String(kw).trim() && String(kw).trim() !== '') {
                const badge = document.createElement('span');
                badge.className = 'keyword-badge';
                badge.textContent = String(kw).trim();
                keywordContainer.appendChild(badge);
                validKeywords++;
            }
        });

        if (validKeywords > 0) {
            keywordCard.style.display = 'block';
            console.log(`✓ Keyword card displayed with ${validKeywords} keywords`);
        } else {
            keywordCard.style.display = 'none';
            console.log("✗ Keyword card hidden - no valid keywords after filtering");
        }
    } else {
        keywordCard.style.display = 'none';
        console.log("✗ Keyword card hidden - no keywords array or empty. Value:", data.keywords);
    }
}

// Display similar news
function displaySimilarNews(similarNews) {
    const container = document.getElementById('similar-news-container');
    container.innerHTML = '';

    if (!similarNews || similarNews.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-inbox"></i>
                <p>Tidak ada berita serupa ditemukan.</p>
            </div>
        `;
        return;
    }

    similarNews.forEach((news, index) => {
        const newsItem = document.createElement('div');
        newsItem.className = 'similar-news-item';
        
        const scorePercent = (news.score * 100).toFixed(1);
        
        newsItem.innerHTML = `
            <div class="similar-news-label">
                <i class="fas fa-tag"></i>
                ${formatCategory(news.label)}
            </div>
            <p class="similar-news-text">${news.text}</p>
            <div class="similar-news-score">
                <i class="fas fa-percentage"></i>
                <span>Tingkat Kemiripan: ${scorePercent}%</span>
            </div>
            ${news.url ? `<button class="btn-view-article" data-url="${news.url}" data-text="${news.text.replace(/"/g, '&quot;')}">
                <i class="fas fa-external-link-alt"></i> Lihat Artikel Lengkap
            </button>` : ''}
        `;
        
        // Add click handler untuk fetch full text
        if (news.url) {
            const viewBtn = newsItem.querySelector('.btn-view-article');
            viewBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                fetchAndAnalyzeFullArticle(news.url, news.text);
            });
        }
        
        container.appendChild(newsItem);
    });
}

// Analyze news
async function analyzeNews() {
    const input = document.getElementById('news-input');
    const text = input.value.trim();

    if (!text) {
        alert('Mohon masukkan teks berita terlebih dahulu!');
        return;
    }

    const analyzeBtn = document.getElementById('analyze-btn');
    const btnText = analyzeBtn.querySelector('.btn-text');
    const btnLoader = analyzeBtn.querySelector('.btn-loader');

    // Show loading state
    analyzeBtn.disabled = true;
    btnText.style.display = 'none';
    btnLoader.style.display = 'inline';

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text })
        });

        if (!response.ok) {
            throw new Error('Terjadi kesalahan saat menganalisis berita');
        }

        const data = await response.json();
        console.log("Response data:", data); // Debug log
        displayResults(data);

    } catch (error) {
        console.error('Error:', error);
        alert('Terjadi kesalahan: ' + error.message);
    } finally {
        // Hide loading state
        analyzeBtn.disabled = false;
        btnText.style.display = 'inline';
        btnLoader.style.display = 'none';
    }
}

// Event Listeners
document.addEventListener('DOMContentLoaded', function() {
    setCurrentDate();

    // Analyze button click
    const analyzeBtn = document.getElementById('analyze-btn');
    analyzeBtn.addEventListener('click', analyzeNews);

    // Character count update
    const newsInput = document.getElementById('news-input');
    newsInput.addEventListener('input', function() {
        updateCharCount();
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
    });

    // Enter key in textarea (Ctrl+Enter to submit)
    newsInput.addEventListener('keydown', function(e) {
        if (e.ctrlKey && e.key === 'Enter') {
            analyzeNews();
        }
    });

    // Initial character count
    updateCharCount();
});

// Fetch full article from URL and analyze
async function fetchAndAnalyzeFullArticle(url, previewText) {
    // Show loading modal
    showLoadingModal('Mengambil artikel lengkap...');
    
    try {
        // Step 1: Fetch full text dari URL
        const fetchResponse = await fetch('/fetch-article', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ url: url })
        });
        
        if (!fetchResponse.ok) {
            throw new Error('Gagal mengambil artikel dari URL');
        }
        
        const fetchData = await fetchResponse.json();
        
        if (!fetchData.success || !fetchData.full_text) {
            throw new Error('Tidak dapat mengambil teks lengkap artikel');
        }
        
        // Step 2: Analisis artikel lengkap
        updateLoadingModal('Menganalisis artikel...');
        
        const analyzeResponse = await fetch('/analyze-article', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                full_text: fetchData.full_text,
                url: url
            })
        });
        
        if (!analyzeResponse.ok) {
            throw new Error('Gagal menganalisis artikel');
        }
        
        const analyzeData = await analyzeResponse.json();
        
        // Step 3: Tampilkan hasil analisis lengkap
        hideLoadingModal();
        displayFullArticleResults(analyzeData, url);
        
    } catch (error) {
        hideLoadingModal();
        console.error('Error:', error);
        alert('Terjadi kesalahan: ' + error.message);
    }
}

// Display full article analysis results
function displayFullArticleResults(data, url) {
    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
    
    // Clear previous results
    const resultsSection = document.getElementById('results-section');
    
    // Update input dengan full text
    document.getElementById('news-input').value = data.input_text;
    updateCharCount();
    
    // Display results dengan semua fitur
    displayResults(data);
    
    // Highlight bahwa ini adalah artikel lengkap
    const categoryCard = resultsSection.querySelector('.result-card');
    if (categoryCard) {
        categoryCard.style.border = '2px solid var(--primary-color)';
        categoryCard.style.boxShadow = '0 0 20px rgba(37, 99, 235, 0.3)';
    }
    
    // Show URL jika ada
    if (url) {
        const urlBadge = document.createElement('div');
        urlBadge.className = 'article-url-badge';
        urlBadge.innerHTML = `
            <i class="fas fa-link"></i>
            <a href="${url}" target="_blank" rel="noopener noreferrer">Buka Artikel Asli</a>
        `;
        
        const categoryCardHeader = resultsSection.querySelector('.result-card .card-header');
        if (categoryCardHeader) {
            categoryCardHeader.appendChild(urlBadge);
        }
    }
}

// Loading modal functions
function showLoadingModal(message) {
    let modal = document.getElementById('loading-modal');
    if (!modal) {
        modal = document.createElement('div');
        modal.id = 'loading-modal';
        modal.className = 'loading-modal';
        modal.innerHTML = `
            <div class="loading-modal-content">
                <i class="fas fa-spinner fa-spin"></i>
                <p id="loading-message">${message}</p>
            </div>
        `;
        document.body.appendChild(modal);
    }
    document.getElementById('loading-message').textContent = message;
    modal.style.display = 'flex';
}

function updateLoadingModal(message) {
    const modal = document.getElementById('loading-modal');
    if (modal) {
        document.getElementById('loading-message').textContent = message;
    }
}

function hideLoadingModal() {
    const modal = document.getElementById('loading-modal');
    if (modal) {
        modal.style.display = 'none';
    }
}

