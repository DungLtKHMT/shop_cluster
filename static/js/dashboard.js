// Dashboard JavaScript

let chartsInitialized = false;
let charts = {};

// Initialize dashboard
async function initDashboard() {
    console.log('Initializing dashboard...');
    await loadOverview();
    await loadRules();
    await loadStrategies();
    populateClusterSelector();
}

// Load overview data
async function loadOverview() {
    try {
        const response = await fetch('/api/overview');
        const data = await response.json();
        
        console.log('Overview data:', data);
        
        // Update stats
        document.getElementById('stat-customers').textContent = data.total_customers || '-';
        document.getElementById('stat-clusters').textContent = data.n_clusters || '-';
        document.getElementById('stat-rules-apriori').textContent = data.total_rules_apriori || '-';
        document.getElementById('stat-rules-fpgrowth').textContent = data.total_rules_fpgrowth || '-';
        
        // Update selected K
        if (data.n_clusters) {
            document.getElementById('selected-k').textContent = data.n_clusters;
        }
        
        // Initialize charts
        if (!chartsInitialized) {
            await initCharts(data);
            chartsInitialized = true;
        }
        
    } catch (error) {
        console.error('Error loading overview:', error);
    }
}

// Initialize all charts
async function initCharts(overviewData) {
    console.log('Initializing charts...');
    
    // 1. Cluster Distribution Chart (Doughnut)
    const clusterDistCtx = document.getElementById('clusterDistChart');
    if (clusterDistCtx && overviewData.cluster_distribution) {
        const distribution = overviewData.cluster_distribution;
        const labels = Object.keys(distribution).map(k => `Cụm ${k}`);
        const values = Object.values(distribution);
        const colors = generateColors(labels.length);
        
        charts.clusterDist = new Chart(clusterDistCtx, {
            type: 'doughnut',
            data: {
                labels: labels,
                datasets: [{
                    data: values,
                    backgroundColor: colors,
                    borderWidth: 2,
                    borderColor: '#fff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            font: {
                                family: 'Inter',
                                size: 12
                            }
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                const percentage = ((context.parsed / total) * 100).toFixed(1);
                                return `${context.label}: ${context.parsed} (${percentage}%)`;
                            }
                        }
                    }
                }
            }
        });
    }
    
    // 2. RFM Charts (3 separate charts for better comparison)
    try {
        const response = await fetch('/api/rfm-comparison');
        const rfmData = await response.json();
        
        if (!rfmData.error && rfmData.length > 0) {
            const labels = rfmData.map(d => `Cụm ${d.cluster_id}`);
            const colors = generateColors(rfmData.length);
            
            // 2a. Recency Chart
            const recencyCtx = document.getElementById('recencyChart');
            if (recencyCtx) {
                charts.recency = new Chart(recencyCtx, {
                    type: 'bar',
                    data: {
                        labels: labels,
                        datasets: [{
                            label: 'Recency (ngày)',
                            data: rfmData.map(d => d.recency_mean),
                            backgroundColor: colors.map(c => c.replace('0.8', '0.7')),
                            borderColor: colors,
                            borderWidth: 2
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: true,
                        scales: {
                            y: {
                                beginAtZero: true,
                                title: {
                                    display: true,
                                    text: 'Số ngày',
                                    font: { family: 'Inter', size: 14, weight: 'bold' }
                                }
                            }
                        },
                        plugins: {
                            legend: { display: false },
                            tooltip: {
                                callbacks: {
                                    label: function(context) {
                                        return `${context.parsed.y.toFixed(1)} ngày (càng thấp càng tốt - mua gần đây)`;
                                    }
                                }
                            }
                        }
                    }
                });
            }
            
            // 2b. Frequency Chart
            const frequencyCtx = document.getElementById('frequencyChart');
            if (frequencyCtx) {
                charts.frequency = new Chart(frequencyCtx, {
                    type: 'bar',
                    data: {
                        labels: labels,
                        datasets: [{
                            label: 'Frequency (đơn)',
                            data: rfmData.map(d => d.frequency_mean),
                            backgroundColor: colors.map(c => c.replace('0.8', '0.7')),
                            borderColor: colors,
                            borderWidth: 2
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: true,
                        scales: {
                            y: {
                                beginAtZero: true,
                                title: {
                                    display: true,
                                    text: 'Số đơn hàng',
                                    font: { family: 'Inter', size: 14, weight: 'bold' }
                                }
                            }
                        },
                        plugins: {
                            legend: { display: false },
                            tooltip: {
                                callbacks: {
                                    label: function(context) {
                                        return `${context.parsed.y.toFixed(1)} đơn (càng cao càng trung thành)`;
                                    }
                                }
                            }
                        }
                    }
                });
            }
            
            // 2c. Monetary Chart
            const monetaryCtx = document.getElementById('monetaryChart');
            if (monetaryCtx) {
                charts.monetary = new Chart(monetaryCtx, {
                    type: 'bar',
                    data: {
                        labels: labels,
                        datasets: [{
                            label: 'Monetary (£)',
                            data: rfmData.map(d => d.monetary_mean),
                            backgroundColor: colors.map(c => c.replace('0.8', '0.7')),
                            borderColor: colors,
                            borderWidth: 2
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: true,
                        scales: {
                            y: {
                                beginAtZero: true,
                                title: {
                                    display: true,
                                    text: 'Tổng chi tiêu (£)',
                                    font: { family: 'Inter', size: 14, weight: 'bold' }
                                }
                            }
                        },
                        plugins: {
                            legend: { display: false },
                            tooltip: {
                                callbacks: {
                                    label: function(context) {
                                        return `£${context.parsed.y.toFixed(2)} (càng cao càng giá trị)`;
                                    }
                                }
                            }
                        }
                    }
                });
            }
        }
    } catch (error) {
        console.error('Error loading RFM data:', error);
    }
    
    // 3. Silhouette Score Chart (Line)
    const silhouetteCtx = document.getElementById('silhouetteChart');
    if (silhouetteCtx) {
        // Real silhouette scores from evaluation
        const kValues = [2, 3, 4, 5, 6, 7, 8, 9, 10];
        const scores = [0.8541, 0.5813, 0.4801, 0.4875, 0.4928, 0.4947, 0.4841, 0.4865, 0.4848];
        
        charts.silhouette = new Chart(silhouetteCtx, {
            type: 'line',
            data: {
                labels: kValues.map(k => `K=${k}`),
                datasets: [{
                    label: 'Silhouette Score',
                    data: scores,
                    borderColor: 'rgb(6, 182, 212)',
                    backgroundColor: 'rgba(6, 182, 212, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 5,
                    pointHoverRadius: 7,
                    pointBackgroundColor: 'rgb(6, 182, 212)',
                    pointBorderColor: '#fff',
                    pointBorderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1,
                        title: {
                            display: true,
                            text: 'Silhouette Score',
                            font: {
                                family: 'Inter',
                                size: 14,
                                weight: 'bold'
                            }
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Số Cụm K',
                            font: {
                                family: 'Inter',
                                size: 14,
                                weight: 'bold'
                            }
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `Score: ${context.parsed.y.toFixed(3)}`;
                            }
                        }
                    }
                }
            }
        });
    }
    
    // 4. PCA Scatter Plot
    const pcaCtx = document.getElementById('pcaScatterChart');
    if (pcaCtx) {
        try {
            const response = await fetch('/api/pca-data');
            const pcaData = await response.json();
            
            if (!pcaData.error && pcaData.clusters) {
                const colors = generateColors(Object.keys(pcaData.clusters).length);
                const datasets = [];
                
                Object.entries(pcaData.clusters).forEach(([clusterId, data], index) => {
                    datasets.push({
                        label: `Cụm ${clusterId}`,
                        data: data.x.map((x, i) => ({x: x, y: data.y[i]})),
                        backgroundColor: colors[index],
                        borderColor: colors[index].replace('0.8', '1'),
                        pointRadius: 3,
                        pointHoverRadius: 5
                    });
                });
                
                charts.pca = new Chart(pcaCtx, {
                    type: 'scatter',
                    data: { datasets: datasets },
                    options: {
                        responsive: true,
                        plugins: {
                            legend: {
                                position: 'top',
                                labels: {
                                    font: {
                                        family: 'Inter',
                                        size: 12
                                    }
                                }
                            },
                            title: {
                                display: true,
                                text: 'Cụm Khách Hàng (PCA 2D)',
                                font: {
                                    family: 'Inter',
                                    size: 16,
                                    weight: 'bold'
                                }
                            },
                            tooltip: {
                                callbacks: {
                                    label: function(context) {
                                        return `${context.dataset.label}: (${context.parsed.x.toFixed(2)}, ${context.parsed.y.toFixed(2)})`;
                                    }
                                }
                            }
                        },
                        scales: {
                            x: {
                                title: {
                                    display: true,
                                    text: 'PC1',
                                    font: {
                                        family: 'Inter',
                                        size: 14,
                                        weight: 'bold'
                                    }
                                }
                            },
                            y: {
                                title: {
                                    display: true,
                                    text: 'PC2',
                                    font: {
                                        family: 'Inter',
                                        size: 14,
                                        weight: 'bold'
                                    }
                                }
                            }
                        }
                    }
                });
            }
        } catch (error) {
            console.error('Error loading PCA data:', error);
        }
    }
}

// Load rules
async function loadRules() {
    const algorithm = document.getElementById('rule-algorithm').value;
    const topN = parseInt(document.getElementById('rule-topn').value);
    const minLift = parseFloat(document.getElementById('rule-minlift').value);
    
    try {
        const response = await fetch(`/api/rules?algorithm=${algorithm}&top_n=${topN}&min_lift=${minLift}`);
        const rules = await response.json();
        
        console.log(`Loaded ${rules.length} rules`);
        
        const tbody = document.getElementById('rules-table-body');
        tbody.innerHTML = '';
        
        if (rules.length === 0) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="6" class="px-6 py-8 text-center text-gray-500">
                        Không tìm thấy luật nào với bộ lọc hiện tại
                    </td>
                </tr>
            `;
            return;
        }
        
        rules.forEach((rule, index) => {
            const row = document.createElement('tr');
            row.className = 'hover:bg-blue-50 transition-colors duration-150';
            
            // Color code lift
            let liftColor = 'text-gray-700';
            if (rule.lift >= 50) liftColor = 'text-red-600 font-bold';
            else if (rule.lift >= 20) liftColor = 'text-orange-600 font-semibold';
            else if (rule.lift >= 10) liftColor = 'text-emerald-600 font-semibold';
            else if (rule.lift >= 5) liftColor = 'text-blue-600';
            
            row.innerHTML = `
                <td class="px-6 py-4 text-gray-600">${index + 1}</td>
                <td class="px-6 py-4 text-gray-700 font-medium">${rule.antecedents}</td>
                <td class="px-6 py-4 text-gray-700 font-medium">${rule.consequents}</td>
                <td class="px-6 py-4 text-center text-gray-600">${rule.support.toFixed(4)}</td>
                <td class="px-6 py-4 text-center text-gray-600">${rule.confidence.toFixed(4)}</td>
                <td class="px-6 py-4 text-center ${liftColor}">${rule.lift.toFixed(2)}</td>
            `;
            
            tbody.appendChild(row);
        });
        
    } catch (error) {
        console.error('Error loading rules:', error);
    }
}

// Populate cluster selector
function populateClusterSelector() {
    const selector = document.getElementById('cluster-selector');
    const nClusters = parseInt(document.getElementById('stat-clusters').textContent);
    
    if (isNaN(nClusters) || nClusters === 0) {
        setTimeout(populateClusterSelector, 500);
        return;
    }
    
    selector.innerHTML = '';
    for (let i = 0; i < nClusters; i++) {
        const option = document.createElement('option');
        option.value = i;
        option.textContent = `Cụm ${i}`;
        selector.appendChild(option);
    }
    
    // Load first cluster by default
    loadClusterProfile();
}

// Load cluster profile
async function loadClusterProfile() {
    const clusterId = document.getElementById('cluster-selector').value;
    if (clusterId === '') return;
    
    try {
        // Fetch basic cluster profile
        const profileResponse = await fetch(`/api/cluster-profile/${clusterId}`);
        const profile = await profileResponse.json();
        
        // Fetch top rules for all clusters
        const rulesResponse = await fetch('/api/top-rules-by-cluster?top_n=10');
        const topRulesData = await rulesResponse.json();
        const clusterRules = topRulesData[clusterId];
        
        console.log('Cluster profile:', profile);
        console.log('Top rules:', clusterRules);
        
        const content = document.getElementById('cluster-profile-content');
        
        let html = `
            <!-- Basic Stats -->
            <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
                <div class="bg-gradient-to-br from-cyan-500 to-blue-600 rounded-lg p-6 text-white shadow-lg">
                    <div class="text-3xl font-bold">${profile.size}</div>
                    <div class="text-sm mt-2 opacity-90">Số Khách Hàng</div>
                </div>
                <div class="bg-gradient-to-br from-blue-500 to-indigo-600 rounded-lg p-6 text-white shadow-lg">
                    <div class="text-3xl font-bold">${profile.percentage}%</div>
                    <div class="text-sm mt-2 opacity-90">Tỷ Lệ</div>
                </div>
                <div class="bg-gradient-to-br from-indigo-500 to-purple-600 rounded-lg p-6 text-white shadow-lg">
                    <div class="text-3xl font-bold">Cụm ${profile.cluster_id}</div>
                    <div class="text-sm mt-2 opacity-90">ID Cụm</div>
                </div>
            </div>
        `;
        
        // RFM Stats
        if (profile.avg_recency !== undefined) {
            html += `
                <div class="bg-blue-50 rounded-lg p-6 mb-6">
                    <h3 class="text-lg font-semibold text-gray-700 mb-4">Thống Kê RFM</h3>
                    <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div class="bg-white rounded-lg p-4 border-l-4 border-blue-500">
                            <div class="text-sm text-gray-600">Recency</div>
                            <div class="text-2xl font-bold text-gray-800 mt-1">${profile.avg_recency.toFixed(1)}</div>
                            <div class="text-xs text-gray-500 mt-1">Trung bình: ${profile.avg_recency.toFixed(1)} ngày</div>
                            <div class="text-xs text-gray-500">Trung vị: ${profile.median_recency.toFixed(1)} ngày</div>
                        </div>
                        <div class="bg-white rounded-lg p-4 border-l-4 border-emerald-500">
                            <div class="text-sm text-gray-600">Frequency</div>
                            <div class="text-2xl font-bold text-gray-800 mt-1">${profile.avg_frequency.toFixed(1)}</div>
                            <div class="text-xs text-gray-500 mt-1">Trung bình: ${profile.avg_frequency.toFixed(1)} đơn</div>
                            <div class="text-xs text-gray-500">Trung vị: ${profile.median_frequency.toFixed(1)} đơn</div>
                        </div>
                        <div class="bg-white rounded-lg p-4 border-l-4 border-orange-500">
                            <div class="text-sm text-gray-600">Monetary</div>
                            <div class="text-2xl font-bold text-gray-800 mt-1">£${profile.avg_monetary.toFixed(0)}</div>
                            <div class="text-xs text-gray-500 mt-1">Trung bình: £${profile.avg_monetary.toFixed(2)}</div>
                            <div class="text-xs text-gray-500">Trung vị: £${profile.median_monetary.toFixed(2)}</div>
                        </div>
                    </div>
                </div>
            `;
        }
        
        // Top Rules from actual calculation (Section 7.3 data)
        if (clusterRules && clusterRules.top_rules && clusterRules.top_rules.length > 0) {
            html += `
                <div class="bg-gray-50 rounded-lg p-6">
                    <h3 class="text-lg font-semibold text-gray-700 mb-4">Top 10 Luật Đặc Trưng (Phân Tích Từ Dữ Liệu Thực)</h3>
                    <div class="text-sm text-gray-600 mb-4">
                        Dữ liệu tính toán từ feature matrix ${clusterRules.n_customers} khách hàng × 200 rules với weighting='lift'
                    </div>
                    <div class="overflow-x-auto">
                        <table class="min-w-full bg-white border border-gray-200 rounded-lg overflow-hidden">
                            <thead class="bg-gradient-to-r from-cyan-600 to-blue-600 text-white">
                                <tr>
                                    <th class="px-4 py-3 text-left text-sm font-semibold">#</th>
                                    <th class="px-4 py-3 text-left text-sm font-semibold">Luật</th>
                                    <th class="px-4 py-3 text-center text-sm font-semibold">% KH Kích Hoạt</th>
                                    <th class="px-4 py-3 text-center text-sm font-semibold">Activation</th>
                                    <th class="px-4 py-3 text-center text-sm font-semibold">Lift</th>
                                    <th class="px-4 py-3 text-center text-sm font-semibold">Conf</th>
                                </tr>
                            </thead>
                            <tbody class="divide-y divide-gray-200">
            `;
            
            clusterRules.top_rules.forEach((rule) => {
                const barWidth = Math.min(rule.pct_activated, 100);
                const activationClass = rule.pct_activated > 50 ? 'text-green-700 font-bold' : 'text-gray-700';
                
                html += `
                    <tr class="hover:bg-blue-50 transition">
                        <td class="px-4 py-3 text-gray-600 font-medium">${rule.rank}</td>
                        <td class="px-4 py-3">
                            <div class="text-sm text-gray-800 font-medium">${rule.antecedents}</div>
                            <div class="text-xs text-gray-500 mt-1">→ ${rule.consequents}</div>
                        </td>
                        <td class="px-4 py-3">
                            <div class="flex items-center justify-center">
                                <div class="w-24 bg-gray-200 rounded-full h-3 mr-2">
                                    <div class="bg-gradient-to-r from-green-400 to-emerald-600 h-3 rounded-full transition-all" style="width: ${barWidth}%"></div>
                                </div>
                                <span class="text-sm ${activationClass}">${rule.pct_activated}%</span>
                            </div>
                            <div class="text-xs text-gray-500 text-center mt-1">${rule.customers_activated}/${clusterRules.n_customers} KH</div>
                        </td>
                        <td class="px-4 py-3 text-center">
                            <span class="text-sm font-semibold text-blue-700">${rule.mean_activation}</span>
                        </td>
                        <td class="px-4 py-3 text-center">
                            <span class="text-sm font-semibold text-purple-700">${rule.lift}</span>
                        </td>
                        <td class="px-4 py-3 text-center">
                            <span class="text-sm text-gray-700">${rule.confidence}%</span>
                        </td>
                    </tr>
                `;
            });
            
            html += `
                            </tbody>
                        </table>
                    </div>
                </div>
            `;
        }
        
        content.innerHTML = html;
        
    } catch (error) {
        console.error('Error loading cluster profile:', error);
    }
}

// Load marketing strategies
async function loadStrategies() {
    try {
        const response = await fetch('/api/marketing-strategies');
        const strategies = await response.json();
        
        console.log('Strategies:', strategies);
        
        const content = document.getElementById('strategies-content');
        let html = '';
        
        const colors = [
            'from-cyan-100 to-blue-100',
            'from-blue-100 to-sky-100',
            'from-sky-100 to-cyan-100',
            'from-teal-100 to-emerald-100',
            'from-emerald-100 to-teal-100',
            'from-cyan-50 to-blue-50'
        ];
        
        Object.entries(strategies).forEach(([clusterId, strategy], index) => {
            const colorClass = colors[index % colors.length];
            
            html += `
                <div class="bg-gradient-to-r ${colorClass} rounded-lg shadow-md p-8 border border-cyan-200">
                    <div class="flex items-start justify-between mb-4">
                        <div>
                            <h3 class="text-2xl font-bold text-gray-800">Cụm ${clusterId}: ${strategy.name_vi}</h3>
                            <p class="text-sm text-gray-600 mt-1">${strategy.name_en}</p>
                        </div>
                        <div class="text-4xl font-bold text-cyan-600 opacity-30">${clusterId}</div>
                    </div>
                    
                    <div class="bg-white bg-opacity-70 rounded-lg p-4 mb-4 border border-cyan-100">
                        <p class="text-sm font-semibold mb-2 text-gray-700">Đặc Điểm Khách Hàng (Persona):</p>
                        <p class="text-gray-800">${strategy.persona}</p>
                    </div>
                    
                    <div class="bg-white bg-opacity-70 rounded-lg p-4 border border-cyan-100">
                        <p class="text-sm font-semibold mb-3 text-gray-700">Chiến Lược Marketing:</p>
                        <ul class="space-y-2">
                            ${strategy.strategies.map(s => `
                                <li class="flex items-start">
                                    <span class="text-cyan-600 font-bold mr-2">•</span>
                                    <span class="text-gray-800">${s}</span>
                                </li>
                            `).join('')}
                        </ul>
                    </div>
                </div>
            `;
        });
        
        content.innerHTML = html;
        
    } catch (error) {
        console.error('Error loading strategies:', error);
    }
}

// Helper function to generate colors
function generateColors(n) {
    const colors = [
        'rgba(6, 182, 212, 0.8)',    // cyan
        'rgba(59, 130, 246, 0.8)',   // blue
        'rgba(99, 102, 241, 0.8)',   // indigo
        'rgba(168, 85, 247, 0.8)',   // purple
        'rgba(236, 72, 153, 0.8)',   // pink
        'rgba(16, 185, 129, 0.8)',   // emerald
        'rgba(245, 158, 11, 0.8)',   // orange
        'rgba(239, 68, 68, 0.8)'     // red
    ];
    return colors.slice(0, n);
}
