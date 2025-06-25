// Group data (should match groups.js)
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
  }
];

// Blog metadata
const blogMetadata = [
  {
    id: 'linear-regression',
    title: "Linear Regression",
    subtitle: "This is the most basic and overlooked in today's machine learning world, when we have advanced stuff, like transformers, RNNs and so much more. But in reality, if you dive deep into any kind of model, it will have linear regression in some form or the other!",
    html: '../blog-pages/linear-regression.html',
    tags: ['classical-ml', 'fundamentals', 'regression']
  },
  {
    id: 'optimizers',
    title: "Optimizers",
    subtitle: "A deep dive into the evolution of optimization algorithms in deep learning.",
    html: '../blog-pages/optimizer-theory.html',
    tags: ['deep-learning', 'optimization', 'neural-networks']
  },
  {
    id: "performance-metrics",
    title: "Performance Metrics",
    subtitle: "Performance metrics tell you if your model actually works.",
    html: "../blog-pages/performance-metrics.html",
    tags: ['classical-ml', 'evaluation', 'metrics']
  }
];

// Get the current group ID from the script tag
function getCurrentGroupId() {
  const script = document.querySelector('script[data-group-id]');
  return script ? script.getAttribute('data-group-id') : null;
}

// Calculate group progress
function calculateGroupProgress(groupId) {
  const group = groups.find(g => g.id === groupId);
  if (!group || group.blogIds.length === 0) return 0;

  let totalProgress = 0;
  group.blogIds.forEach(blogId => {
    const progress = parseFloat(localStorage.getItem(`zeus-progress-${blogId}`) || 0);
    totalProgress += progress;
  });

  return parseFloat((totalProgress / group.blogIds.length).toFixed(2));
}

// Load lessons for the current group
function loadGroupLessons(groupId) {
  const group = groups.find(g => g.id === groupId);
  if (!group) return;

  const lessonsList = document.getElementById('lessons-list');
  if (!lessonsList) return;

  lessonsList.innerHTML = '';

  group.blogIds.forEach(blogId => {
    const blog = blogMetadata.find(b => b.id === blogId);
    if (!blog) return;

    const rawProgress = parseFloat(localStorage.getItem(`zeus-progress-${blogId}`) || 0);
    const progress = parseFloat(rawProgress.toFixed(2));
    const isCompleted = progress === 100;

    const card = document.createElement('div');
    card.className = 'card p-8 flex flex-col items-center fadein';

    card.innerHTML = `
      <div class="flex-grow">
        <h3 class="text-2xl font-bold mb-2 text-center text-[#A68A6D] serif">${blog.title}</h3>
        <p class="text-[#4A3E31] mb-4 outfit text-center">${blog.subtitle}</p>
      </div>
      <div class="w-full px-4 mb-4">
        <div class="flex items-center justify-between mb-2">
          <span class="text-sm text-[#4A3E31] outfit">Progress</span>
        </div>
        <div style="background-color: #F0E6D2; border-radius: 9999px; height: 6px;">
          <div style="width: ${progress}%; background-color: ${group.color}; height: 100%; border-radius: 9999px; transition: width 0.3s ease-in-out;"></div>
        </div>
        ${isCompleted ? '<div class="text-xs text-green-600 outfit mt-1 text-center">✓ Completed</div>' : ''}
      </div>
      <a href="${blog.html}" class="inline-block px-6 py-2 rounded-full font-bold text-md bg-[#D7BFAE] hover:bg-[#CBB59A] text-[#3B3025] transition-all duration-200 border border-[#E8D9C3] outfit">
        ${isCompleted ? 'Review Lesson' : 'Start Lesson'}
      </a>
    `;

    lessonsList.appendChild(card);
  });

  // Fade-in animation for cards
  const cards = document.querySelectorAll('.fadein');
  cards.forEach((card, i) => {
    setTimeout(() => card.classList.add('visible'), 200 + i * 150);
  });
}

// Update overall progress display
function updateOverallProgress(groupId) {
  const progress = calculateGroupProgress(groupId);
  const progressBar = document.getElementById('overall-progress');
  const progressText = document.getElementById('progress-text');

  if (progressBar && progressText) {
    // Animate progress bar
    setTimeout(() => {
      progressBar.style.width = `${progress}%`;
      progressText.textContent = `In Progress`;
    }, 500);
  }
}

// Initialize the group detail page
function initializeGroupDetail() {
  const groupId = getCurrentGroupId();
  if (!groupId) {
    console.error('No group ID found');
    return;
  }

  loadGroupLessons(groupId);
  updateOverallProgress(groupId);
}

// Listen for storage changes to update progress in real-time
window.addEventListener('storage', function(e) {
  if (e.key && e.key.startsWith('zeus-progress-')) {
    const groupId = getCurrentGroupId();
    if (groupId) {
      updateOverallProgress(groupId);
      loadGroupLessons(groupId);
    }
  }
});

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', initializeGroupDetail);
