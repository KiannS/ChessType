// Toggle Chess.com date inputs
document.querySelectorAll('input[name="platform"]').forEach(radio => {
    radio.addEventListener('change', (e) => {
        const chesscomDates = document.getElementById('chesscom-dates');
        if (e.target.value === 'chesscom') {
            chesscomDates.style.display = 'block';
        } else {
            chesscomDates.style.display = 'none';
        }
    });
});

// Form submission
document.getElementById('analysis-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const platform = document.querySelector('input[name="platform"]:checked').value;
    const username = document.getElementById('username').value.trim();
    const maxGames = parseInt(document.getElementById('max-games').value);
    
    if (!username) {
        showError('Please enter a username');
        return;
    }
    
    // Show loading
    document.getElementById('input-section').style.display = 'none';
    document.getElementById('results-section').style.display = 'none';
    document.getElementById('error-section').style.display = 'none';
    document.getElementById('loading-section').style.display = 'block';
    
    try {
        let response;
        
        if (platform === 'lichess') {
            response = await fetch('/analyze-lichess', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    username: username,
                    max_games: maxGames
                })
            });
        } else {
            const year = parseInt(document.getElementById('year').value);
            const month = parseInt(document.getElementById('month').value);
            
            response = await fetch('/analyze-chesscom', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    username: username,
                    year: year,
                    month: month,
                    max_games: maxGames
                })
            });
        }
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Analysis failed');
        }
        
        const data = await response.json();
        displayResults(data);
        
    } catch (error) {
        showError(error.message);
    }
});

function displayResults(data) {
    // Hide loading
    document.getElementById('loading-section').style.display = 'none';
    document.getElementById('results-section').style.display = 'block';
    
    // Style Analysis
    document.getElementById('style-badge').textContent = data.style_analysis.predicted_style;
    document.getElementById('confidence').textContent = 
        `${(data.style_analysis.confidence * 100).toFixed(0)}% confidence`;
    document.getElementById('description').textContent = data.style_analysis.description;
    
    // Strengths
    const strengthsList = document.getElementById('strengths-list');
    strengthsList.innerHTML = '';
    data.style_analysis.strengths.forEach(strength => {
        const li = document.createElement('li');
        li.textContent = strength;
        strengthsList.appendChild(li);
    });
    
    // Famous Players
    document.getElementById('famous-players').textContent = 
        data.style_analysis.famous_players.join(', ');
    
    // Position Performance
    const positionPerf = document.getElementById('position-performance');
    positionPerf.innerHTML = '';
    
    if (data.position_analysis.performance_by_type) {
        Object.entries(data.position_analysis.performance_by_type).forEach(([type, stats]) => {
            const div = document.createElement('div');
            div.className = `position-item ${stats.rating.toLowerCase()}`;
            div.innerHTML = `
                <div>
                    <div class="position-name">${type}</div>
                    <span class="rating-badge ${stats.rating.toLowerCase()}">${stats.rating}</span>
                </div>
                <div class="position-stats">
                    <div class="win-rate">${(stats.win_rate * 100).toFixed(0)}%</div>
                    <small>${stats.positions_encountered} positions</small>
                </div>
            `;
            positionPerf.appendChild(div);
        });
    }
    
    // Recommendations
    const openingsList = document.getElementById('openings-list');
    openingsList.innerHTML = '';
    data.recommendations.openings.forEach(opening => {
        const li = document.createElement('li');
        li.textContent = opening;
        openingsList.appendChild(li);
    });
    
    const tipsList = document.getElementById('tips-list');
    tipsList.innerHTML = '';
    data.recommendations.improvement_tips.forEach(tip => {
        const li = document.createElement('li');
        li.textContent = tip;
        tipsList.appendChild(li);
    });
    
    const focusList = document.getElementById('focus-list');
    focusList.innerHTML = '';
    data.recommendations.training_focus.forEach(focus => {
        const li = document.createElement('li');
        li.textContent = focus;
        focusList.appendChild(li);
    });
    
    // Statistics
    const statsGrid = document.getElementById('stats-grid');
    statsGrid.innerHTML = `
        <div class="stat-item">
            <span class="stat-value">${data.games_analyzed}</span>
            <span class="stat-label">Games Analyzed</span>
        </div>
        <div class="stat-item">
            <span class="stat-value">${data.statistics.avg_game_length.toFixed(0)}</span>
            <span class="stat-label">Avg Game Length</span>
        </div>
        <div class="stat-item">
            <span class="stat-value">${data.statistics.avg_captures_per_game.toFixed(1)}</span>
            <span class="stat-label">Avg Captures/Game</span>
        </div>
        <div class="stat-item">
            <span class="stat-value">${data.statistics.early_castling_rate.toFixed(0)}%</span>
            <span class="stat-label">Early Castling Rate</span>
        </div>
    `;
}

function showError(message) {
    document.getElementById('loading-section').style.display = 'none';
    document.getElementById('error-section').style.display = 'block';
    document.getElementById('error-message').textContent = message;
}

function resetForm() {
    document.getElementById('input-section').style.display = 'block';
    document.getElementById('results-section').style.display = 'none';
    document.getElementById('error-section').style.display = 'none';
    document.getElementById('loading-section').style.display = 'none';
    document.getElementById('analysis-form').reset();
}