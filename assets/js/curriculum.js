const blogs = [
  {
    id: 'linear-regression',
    title: "Linear Regression",
    subtitle: "This is the most basic and overlooked in today's machine learning world, when we have advanced stuff, like transformers, RNNs and so much more. But in reality, if you dive deep into any kind of model, it will have linear regression in some form or the other!",
    html: 'blog-pages/linear-regression.html'
  },
  {
    id: 'optimizers',
    title: "Optimizers",
    subtitle: "A deep dive into the evolution of optimization algorithms in deep learning.",
    html: 'blog-pages/optimizer-theory.html'
  },
  {
    id: "performance-metrics",
    title: "Performance Metrics",
    subtitle: "Performance metrics tell you if your model actually works.",
    html: "blog-pages/performance-metrics.html",
  }
];

function loadBlogs(blogsToLoad) {
  const blogList = document.getElementById('blog-list');
  blogList.innerHTML = ''; // Clear existing blogs

  blogsToLoad.forEach(blog => {
    const card = document.createElement('div');
    card.className = 'card p-8 flex flex-col items-center fadein';
    const progress = localStorage.getItem(`zeus-progress-${blog.id}`) || 0;
    card.innerHTML = `
      <div class="flex-grow">
        <h2 class="text-2xl font-bold mb-2 text-center text-[#A68A6D] serif">${blog.title}</h2>
        <p class="text-[#4A3E31] mb-4 outfit text-center">${blog.subtitle}</p>
      </div>
      <div class="w-full px-4 mb-4">
        <div style="background-color: #F0E6D2; border-radius: 9999px; height: 6px;">
          <div style="width: ${progress}%; background-color: #A68A6D; height: 100%; border-radius: 9999px; transition: width 0.3s ease-in-out;"></div>
        </div>
      </div>
      <a href="${blog.html}" class="inline-block px-6 py-2 rounded-full font-bold text-md bg-[#D7BFAE] hover:bg-[#CBB59A] text-[#3B3025] transition-all duration-200 border border-[#E8D9C3]">View Lesson</a>
    `;
    blogList.appendChild(card);
  });

  // Fade-in animation for cards
  const cards = document.querySelectorAll('.fadein');
  cards.forEach((card, i) => {
    setTimeout(() => card.classList.add('visible'), 200 + i * 150);
  });
}

loadBlogs(blogs);

const searchBar = document.getElementById('search-bar');
searchBar.addEventListener('keyup', (e) => {
    const searchString = e.target.value.toLowerCase();
    const filteredBlogs = blogs.filter(blog => {
        return (
            blog.title.toLowerCase().includes(searchString) ||
            blog.subtitle.toLowerCase().includes(searchString)
        );
    });
    loadBlogs(filteredBlogs);
});
