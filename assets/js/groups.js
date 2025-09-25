const groups = [
  {
    id: 'classical-ml',
    title: "Classical ML",
    subtitle: "Master the fundamentals of machine learning with traditional algorithms like linear regression, SVMs, and decision trees. Build a solid foundation before diving into deep learning.",
    description: "Explore the core principles and mathematics behind classical machine learning algorithms.",
    topics: ["Linear Regression", "Logistic Regression", "Decision Trees", "Random Forest", "SVM", "K-Means", "Performance Metrics"],
    blogIds: ['linear-regression', 'performance-metrics'],
    color: "#A68A6D"
  },
  {
    id: 'deep-learning',
    title: "Deep Learning",
    subtitle: "Dive deep into neural networks, from basic perceptrons to advanced architectures. Learn about CNNs, RNNs, and the latest breakthroughs in deep learning.",
    description: "Understand neural networks and their applications in solving complex problems.",
    topics: ["Neural Networks", "Backpropagation", "CNNs", "RNNs", "LSTMs", "Optimizers", "Regularization"],
    blogIds: ['optimizers'],
    color: "#A68A6D"
  },
  {
    id: 'nlp',
    title: "Natural Language Processing",
    subtitle: "Process and understand human language with computers. From text preprocessing to transformers, explore how machines comprehend and generate text.",
    description: "Learn to build systems that can understand, interpret, and generate human language.",
    topics: ["Text Preprocessing", "Word Embeddings", "Sentiment Analysis", "Named Entity Recognition", "Transformers", "BERT", "GPT"],
    blogIds: [],
    color: "#A68A6D"
  },
  {
    id: 'computer-vision',
    title: "Computer Vision",
    subtitle: "Teach machines to see and understand visual content. Learn image processing, object detection, and how to build AI systems that can analyze visual data.",
    description: "Develop skills in image processing, pattern recognition, and visual AI applications.",
    topics: ["Image Processing", "Feature Detection", "Object Detection", "Image Classification", "CNNs", "YOLO", "Face Recognition"],
    blogIds: [],
    color: "#A68A6D"
  },
  {
    id: 'reinforcement-learning',
    title: "Reinforcement Learning",
    subtitle: "Train agents to make decisions through trial and error. Explore how AI systems learn optimal strategies in dynamic environments.",
    description: "Master the art of training AI agents to make sequential decisions and learn from experience.",
    topics: ["Q-Learning", "Policy Gradients", "Actor-Critic", "Deep Q-Networks", "Multi-Agent RL", "Game Theory"],
    blogIds: [],
    color: "#A68A6D"
  },
  {
    id: 'mlops',
    title: "MLOps & Deployment",
    subtitle: "Bridge the gap between development and production. Learn to deploy, monitor, and maintain ML models in real-world applications.",
    description: "Learn the practical skills needed to deploy and maintain ML systems in production.",
    topics: ["Model Deployment", "Docker", "CI/CD", "Model Monitoring", "A/B Testing", "Data Pipelines", "Cloud Platforms"],
    blogIds: [],
    color: "#A68A6D"
  }
];

// Blog metadata with tags for filtering
const blogMetadata = [
  {
    id: 'linear-regression',
    title: "Linear Regression",
    subtitle: "This is the most basic and overlooked in today's machine learning world, when we have advanced stuff, like transformers, RNNs and so much more. But in reality, if you dive deep into any kind of model, it will have linear regression in some form or the other!",
    html: 'blog-pages/linear-regression.html',
    tags: ['classical-ml', 'fundamentals', 'regression']
  },
  {
    id: 'optimizers',
    title: "Optimizers",
    subtitle: "A deep dive into the evolution of optimization algorithms in deep learning.",
    html: 'blog-pages/optimizer-theory.html',
    tags: ['deep-learning', 'optimization', 'neural-networks']
  },
  {
    id: "performance-metrics",
    title: "Performance Metrics",
    subtitle: "Performance metrics tell you if your model actually works.",
    html: "blog-pages/performance-metrics.html",
    tags: ['classical-ml', 'evaluation', 'metrics']
  }
];

function getGroupProgress(groupId) {
  const group = groups.find(g => g.id === groupId);
  if (!group || group.blogIds.length === 0) return 0;

  let totalProgress = 0;
  group.blogIds.forEach(blogId => {
    const progress = parseFloat(localStorage.getItem(`zeus-progress-${blogId}`) || 0);
    totalProgress += progress;
  });

  return parseFloat((totalProgress / group.blogIds.length).toFixed(2));
}

function loadGroups(groupsToLoad) {
  const groupsList = document.getElementById('groups-list');
  groupsList.innerHTML = ''; // Clear existing groups

  groupsToLoad.forEach(group => {
    const card = document.createElement('div');
    card.className = 'card p-8 flex flex-col items-center fadein cursor-pointer group-card';
    const progress = getGroupProgress(group.id);
    const blogCount = group.blogIds.length;

    card.innerHTML = `
      <div class="flex-grow text-center">
        <h2 class="text-2xl font-bold mb-2 text-[#A68A6D] serif">${group.title}</h2>
        <p class="text-[#4A3E31] mb-4 outfit">${group.subtitle}</p>
        <div class="mb-4">
          <div class="text-sm text-[#4A3E31] mb-2 outfit">
            ${blogCount} lesson${blogCount !== 1 ? 's' : ''} available
          </div>
          ${blogCount > 0 ? `
            <div class="w-full px-4 mb-2">
              <div style="background-color: #F0E6D2; border-radius: 9999px; height: 6px;">
                <div style="width: ${progress}%; background-color: ${group.color}; height: 100%; border-radius: 9999px; transition: width 0.3s ease-in-out;"></div>
              </div>
            </div>
          ` : `
            <div class="text-xs text-[#A68A6D] outfit italic">Coming Soon</div>
          `}
        </div>
        <div class="flex flex-wrap gap-1 justify-center mb-4">
          ${group.topics.slice(0, 4).map(topic =>
            `<span class="text-xs px-2 py-1 rounded-full bg-[#F0E6D2] text-[#4A3E31] outfit">${topic}</span>`
          ).join('')}
          ${group.topics.length > 4 ? `<span class="text-xs px-2 py-1 rounded-full bg-[#F0E6D2] text-[#4A3E31] outfit">+${group.topics.length - 4} more</span>` : ''}
        </div>
      </div>
      <button class="inline-block px-6 py-2 rounded-full font-bold text-md bg-[#D7BFAE] hover:bg-[#CBB59A] text-[#3B3025] transition-all duration-200 border border-[#E8D9C3] outfit group-btn">
        ${blogCount > 0 ? 'Explore Group' : 'Coming Soon'}
      </button>
    `;

    if (blogCount > 0) {
      card.addEventListener('click', () => {
        window.location.href = `/pages/groups/${group.id}.html`;
      });
    } else {
      card.style.opacity = '0.7';
      card.style.cursor = 'default';
    }

    groupsList.appendChild(card);
  });

  // Fade-in animation for cards
  const cards = document.querySelectorAll('.fadein');
  cards.forEach((card, i) => {
    setTimeout(() => card.classList.add('visible'), 200 + i * 150);
  });
}

loadGroups(groups);

const searchBar = document.getElementById('search-bar');
searchBar.addEventListener('keyup', (e) => {
    const searchString = e.target.value.toLowerCase();
    const filteredGroups = groups.filter(group => {
        return (
            group.title.toLowerCase().includes(searchString) ||
            group.subtitle.toLowerCase().includes(searchString) ||
            group.topics.some(topic => topic.toLowerCase().includes(searchString))
        );
    });
    loadGroups(filteredGroups);
});
