// Fade-in animation for cards
document.addEventListener('DOMContentLoaded', () => {
  const cards = document.querySelectorAll('.fadein');
  cards.forEach((card, i) => {
    setTimeout(() => card.classList.add('visible'), 200 + i * 150);
  });
}); 