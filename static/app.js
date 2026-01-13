document.getElementById('analyzeForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const platform = document.getElementById('platform').value;
    const username = document.getElementById('username').value;
    
    // Hide previous results/errors
    document.getElementById('results').style.display = 'none';
    document.getElementById('error').style.display = 'none';
    
    // Show loading state
    const btnText = document.getElementById('btnText');
    const btnLoader = document.getElementById('btnLoader');
    const analyzeBtn = document.getElementById('analyzeBtn');
    
    btnText.style.display = 'none';
    btnLoader.style.display = 'inline-block';
    analyzeBtn.disabled = true;
    
    try {
        let endpoint, requestData;
        
        if (platform === 'lichess') {
            endpoint = '/analyze-lichess';
            requestData = {
                username: username,
                max_games: 1000  // Set high limit for all games
            };
        } else {
            endpoint = '/analyze-chesscom';
            requestData = {
                username: username,
                max_games: 1000  // Set high limit for all games
            };
        }
        
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });
        
        const data = await response.json();
        
        if (data.error) {
            showError(data.error);
        } else {
            displayResults(data);
        }
    } catch (error) {
        showError('An error occurred while analyzing your games. Please try again.');
        console.error('Error:', error);
    } finally {
        // Reset button state
        btnText.style.display = 'inline';
        btnLoader.style.display = 'none';
        analyzeBtn.disabled = false;
    }
});

function showError(message) {
    document.getElementById('errorMessage').textContent = message;
    document.getElementById('error').style.display = 'block';
    window.scrollTo({ top: document.getElementById('error').offsetTop - 20, behavior: 'smooth' });
}

function displayResults(data) {
    document.getElementById('resultUsername').textContent = data.username;
    document.getElementById('gamesCount').textContent = data.games_analyzed;
    
    // Style Analysis
    const styleBadge = document.getElementById('styleBadge');
    styleBadge.textContent = data.style_analysis.predicted_style;
    styleBadge.className = 'style-badge ' + data.style_analysis.predicted_style.toLowerCase();
    
    document.getElementById('confidence').textContent = 
        (data.style_analysis.confidence * 100).toFixed(1) + '%';
    document.getElementById('styleDescription').textContent = 
        data.style_analysis.description;
    
    // Strengths
    const strengthsList = document.getElementById('strengths');
    strengthsList.innerHTML = '';
    data.style_analysis.strengths.forEach(strength => {
        const li = document.createElement('li');
        li.textContent = strength;
        strengthsList.appendChild(li);
    });
    
    // Famous Players
    const famousPlayersList = document.getElementById('famousPlayers');
    famousPlayersList.innerHTML = '';
    data.style_analysis.famous_players.forEach(player => {
        const li = document.createElement('li');
        li.textContent = player;
        famousPlayersList.appendChild(li);
    });
    
    // Style Probabilities - FIXED: Now creates proper HTML structure
    const styleProbs = document.getElementById('styleProbs');
    styleProbs.innerHTML = '';
    
    // Sort by probability descending
    const sortedProbs = Object.entries(data.style_analysis.style_probabilities)
        .sort((a, b) => b[1] - a[1]);
    
    sortedProbs.forEach(([style, prob]) => {
        const barWrapper = document.createElement('div');
        barWrapper.className = 'prob-bar';
        
        const label = document.createElement('span');
        label.className = 'prob-label';
        label.textContent = style;
        
        const fillContainer = document.createElement('div');
        fillContainer.className = 'prob-fill-container';
        
        const fill = document.createElement('div');
        fill.className = 'prob-fill';
        fill.style.width = (prob * 100) + '%';
        
        const value = document.createElement('span');
        value.className = 'prob-value';
        value.textContent = (prob * 100).toFixed(1) + '%';
        
        fillContainer.appendChild(fill);
        barWrapper.appendChild(label);
        barWrapper.appendChild(fillContainer);
        barWrapper.appendChild(value);
        styleProbs.appendChild(barWrapper);
    });
    
    // Position Analysis
    if (data.position_analysis.best_position_type) {
        document.getElementById('bestPosition').textContent = 
            data.position_analysis.best_position_type.type;
        document.getElementById('bestPositionWinRate').textContent = 
            (data.position_analysis.best_position_type.win_rate * 100).toFixed(1) + '% win rate';
    }
    
    if (data.position_analysis.worst_position_type) {
        document.getElementById('worstPosition').textContent = 
            data.position_analysis.worst_position_type.type;
        document.getElementById('worstPositionWinRate').textContent = 
            (data.position_analysis.worst_position_type.win_rate * 100).toFixed(1) + '% win rate';
    }
    
    // Performance by Position Type - FIXED: Now creates proper card structure
    const positionPerf = document.getElementById('positionPerformance');
    positionPerf.innerHTML = '';
    
    // Sort by win rate descending
    const sortedPerf = Object.entries(data.position_analysis.performance_by_type)
        .sort((a, b) => b[1].win_rate - a[1].win_rate);
    
    sortedPerf.forEach(([posType, stats]) => {
        const card = document.createElement('div');
        card.className = 'position-card ' + stats.rating.toLowerCase();
        
        const typeDiv = document.createElement('div');
        typeDiv.className = 'position-type';
        typeDiv.textContent = posType;
        
        const ratingDiv = document.createElement('div');
        ratingDiv.className = 'position-rating';
        ratingDiv.textContent = stats.rating;
        
        const statsDiv = document.createElement('div');
        statsDiv.className = 'position-stats';
        statsDiv.innerHTML = `Win Rate: ${(stats.win_rate * 100).toFixed(1)}%<br>Positions: ${stats.positions_encountered}`;
        
        card.appendChild(typeDiv);
        card.appendChild(ratingDiv);
        card.appendChild(statsDiv);
        positionPerf.appendChild(card);
    });
    
    // Position Distribution - FIXED: Now creates proper bars
    const positionDist = document.getElementById('positionDistribution');
    positionDist.innerHTML = '';
    
    // Sort by percentage descending
    const sortedDist = Object.entries(data.position_analysis.position_distribution)
        .sort((a, b) => b[1] - a[1]);
    
    sortedDist.forEach(([posType, percentage]) => {
        const barWrapper = document.createElement('div');
        barWrapper.className = 'prob-bar';
        
        const label = document.createElement('span');
        label.className = 'prob-label';
        label.textContent = posType;
        
        const fillContainer = document.createElement('div');
        fillContainer.className = 'prob-fill-container';
        
        const fill = document.createElement('div');
        fill.className = 'prob-fill';
        fill.style.width = percentage + '%';
        
        const value = document.createElement('span');
        value.className = 'prob-value';
        value.textContent = percentage.toFixed(1) + '%';
        
        fillContainer.appendChild(fill);
        barWrapper.appendChild(label);
        barWrapper.appendChild(fillContainer);
        barWrapper.appendChild(value);
        positionDist.appendChild(barWrapper);
    });
    
    // Recommendations
    const recommendations = document.getElementById('recommendations');
    recommendations.innerHTML = '';
    
    if (data.recommendations.training_focus) {
        const focusSection = document.createElement('div');
        focusSection.className = 'subsection';
        focusSection.innerHTML = '<h4>Training Focus</h4>';
        const focusList = document.createElement('ul');
        data.recommendations.training_focus.forEach(item => {
            const li = document.createElement('li');
            li.textContent = item;
            focusList.appendChild(li);
        });
        focusSection.appendChild(focusList);
        recommendations.appendChild(focusSection);
    }
    
    if (data.recommendations.improvement_tips) {
        const tipsSection = document.createElement('div');
        tipsSection.className = 'subsection';
        tipsSection.innerHTML = '<h4>Improvement Tips</h4>';
        const tipsList = document.createElement('ul');
        data.recommendations.improvement_tips.forEach(item => {
            const li = document.createElement('li');
            li.textContent = item;
            tipsList.appendChild(li);
        });
        tipsSection.appendChild(tipsList);
        recommendations.appendChild(tipsSection);
    }
    
    if (data.recommendations.openings) {
        const openingsSection = document.createElement('div');
        openingsSection.className = 'subsection';
        openingsSection.innerHTML = '<h4>Recommended Openings</h4>';
        const openingsList = document.createElement('ul');
        data.recommendations.openings.forEach(item => {
            const li = document.createElement('li');
            li.textContent = item;
            openingsList.appendChild(li);
        });
        openingsSection.appendChild(openingsList);
        recommendations.appendChild(openingsSection);
    }
    
    // Statistics
    const statistics = document.getElementById('statistics');
    statistics.innerHTML = `
        <div class="stats-grid">
            <div class="stat-item">
                <div class="stat-label">Total Moves Analyzed</div>
                <div class="stat-value">${data.statistics.total_moves_analyzed}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Avg Game Length</div>
                <div class="stat-value">${data.statistics.avg_game_length.toFixed(1)} moves</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Avg Captures per Game</div>
                <div class="stat-value">${data.statistics.avg_captures_per_game.toFixed(1)}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Early Castling Rate</div>
                <div class="stat-value">${data.statistics.early_castling_rate.toFixed(1)}%</div>
            </div>
        </div>
    `;
    
    if (data.statistics.favorite_openings && data.statistics.favorite_openings.length > 0) {
        const openingsDiv = document.createElement('div');
        openingsDiv.className = 'subsection';
        openingsDiv.innerHTML = '<h4>Your Favorite Openings</h4>';
        const openingsList = document.createElement('ul');
        data.statistics.favorite_openings.forEach(opening => {
            const li = document.createElement('li');
            li.textContent = opening;
            openingsList.appendChild(li);
        });
        openingsDiv.appendChild(openingsList);
        statistics.appendChild(openingsDiv);
    }
    
    // Show results
    document.getElementById('results').style.display = 'block';
    window.scrollTo({ top: document.getElementById('results').offsetTop - 20, behavior: 'smooth' });
}