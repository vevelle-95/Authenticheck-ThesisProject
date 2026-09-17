const panel = document.querySelector('#authPanel');
const trigger = document.querySelector('#floatingTrigger');
const dim = document.querySelector('#pageDim');
const closeButton = document.querySelector('#closePanel');
const tabs = [...document.querySelectorAll('.tab')];
const pages = [...document.querySelectorAll('.tab-page')];
const qualityItems = [...document.querySelectorAll('.quality-item')];
const filterLabel = document.querySelector('#filterLabel');
const toast = document.querySelector('#toast');

function openPanel() {
  panel.classList.add('open');
  panel.setAttribute('aria-hidden', 'false');
  trigger.classList.add('hidden');
  dim.classList.add('visible');
  closeButton.focus();
}

function closePanel() {
  panel.classList.remove('open');
  panel.setAttribute('aria-hidden', 'true');
  trigger.classList.remove('hidden');
  dim.classList.remove('visible');
  trigger.focus();
}

function showTab(name) {
  tabs.forEach(tab => tab.classList.toggle('active', tab.dataset.tab === name));
  pages.forEach(page => page.classList.toggle('active', page.dataset.page === name));
  panel.querySelector('.panel-scroll').scrollTo({ top: 278, behavior: 'smooth' });
}

trigger.addEventListener('click', openPanel);
closeButton.addEventListener('click', closePanel);
dim.addEventListener('click', closePanel);
document.addEventListener('keydown', event => {
  if (event.key === 'Escape' && panel.classList.contains('open')) closePanel();
});

tabs.forEach(tab => tab.addEventListener('click', () => showTab(tab.dataset.tab)));

qualityItems.forEach(item => item.addEventListener('click', () => {
  qualityItems.forEach(other => other.classList.remove('selected'));
  item.classList.add('selected');
  const label = item.querySelector('strong').textContent.toLowerCase();
  filterLabel.textContent = `Showing ${label} reviews`;
  showTab('reviews');
}));

document.querySelectorAll('.choice').forEach(choice => choice.addEventListener('click', () => {
  [...choice.parentElement.children].forEach(sibling => sibling.classList.remove('active'));
  choice.classList.add('active');
}));

document.querySelector('#marketToggle').addEventListener('click', () => {
  const isLazada = document.body.classList.toggle('lazada');
  document.querySelector('#marketName').textContent = isLazada ? 'LazMall' : 'Shopmall';
});

document.querySelector('#infoButton').addEventListener('click', () => {
  showTab('overview');
  toast.querySelector('p').textContent = 'Authentic review share is the portion classified as authentic.';
  toast.classList.add('show');
  setTimeout(() => toast.classList.remove('show'), 2600);
});

document.querySelector('#filterButton').addEventListener('click', () => {
  filterLabel.textContent = filterLabel.textContent.includes('all') ? 'Showing authentic reviews' : 'Showing all classifications';
});

document.querySelector('#feedbackButton').addEventListener('click', () => {
  toast.querySelector('p').textContent = 'Thanks for your feedback!';
  toast.classList.add('show');
  setTimeout(() => toast.classList.remove('show'), 2200);
});

document.querySelectorAll('.aspect').forEach(aspect => aspect.addEventListener('click', () => {
  aspect.classList.toggle('expanded');
}));

setTimeout(() => trigger.classList.add('attention'), 900);
