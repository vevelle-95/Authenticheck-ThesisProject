const panel = document.querySelector('#authPanel');
const trigger = document.querySelector('#floatingTrigger');
const dim = document.querySelector('#pageDim');
const closeButton = document.querySelector('#closePanel');
const tabs = [...document.querySelectorAll('.tab')];
const pages = [...document.querySelectorAll('.tab-page')];
const qualityItems = [...document.querySelectorAll('.quality-item')];
const filterLabel = document.querySelector('#filterLabel');
const toast = document.querySelector('#toast');
const panelScroll = document.querySelector('#panelScroll');
const loadingStage = document.querySelector('#loadingStage');
const loadingProgress = document.querySelector('#loadingProgress');
const loadingTip = document.querySelector('#loadingTip');
const rerunButton = document.querySelector('#rerunButton');
const triggerTitle = trigger.querySelector('.trigger-copy strong');
const triggerSubtitle = trigger.querySelector('.trigger-copy small');
const triggerScore = trigger.querySelector('.trigger-score');

const demoTips = [
  'Hindi lahat ng reviews ay dapat paniwalaan.',
  'Mas kapaki-pakinabang ang reviews na may tiyak na detalye.',
  'Tinitingnan din kung tugma ang rating at review text.',
  'Buyer photos can provide useful product context.',
  'Delivery comments may not describe the product itself.'
];
const demoSteps = [
  { delay: 0, label: 'Collecting 2,431 visible product reviews', progress: 18 },
  { delay: 1200, label: 'Checking review-quality signals', progress: 44 },
  { delay: 2400, label: 'Comparing text, ratings, and buyer media', progress: 68 },
  { delay: 3600, label: 'Analyzing authentic-review sentiment', progress: 86 },
  { delay: 4600, label: 'Preparing the review intelligence summary', progress: 96 }
];
let hasAnalyzed = false;
let demoRun = 0;
let demoTimers = [];

function openPanel() {
  panel.classList.add('open');
  panel.setAttribute('aria-hidden', 'false');
  trigger.classList.add('hidden');
  dim.classList.add('visible');
  closeButton.focus();
  if (!hasAnalyzed) runDemoAnalysis();
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

rerunButton.addEventListener('click', runDemoAnalysis);

document.querySelectorAll('.aspect').forEach(aspect => aspect.addEventListener('click', () => {
  aspect.classList.toggle('expanded');
}));

setTimeout(() => trigger.classList.add('attention'), 900);

function clearDemoTimers() {
  demoTimers.forEach(timer => clearTimeout(timer));
  demoTimers = [];
}

function runDemoAnalysis() {
  const runId = ++demoRun;
  clearDemoTimers();
  hasAnalyzed = true;
  panelScroll.classList.add('is-analyzing');
  panelScroll.scrollTo({ top: 0 });
  rerunButton.disabled = true;
  rerunButton.textContent = 'Analyzing…';
  loadingProgress.style.width = '4%';
  triggerTitle.textContent = 'Analyzing reviews…';
  triggerSubtitle.textContent = 'Preparing the illustrative demo';
  triggerScore.textContent = '';
  triggerScore.classList.add('loading');

  demoSteps.forEach((step, index) => {
    demoTimers.push(setTimeout(() => {
      if (runId !== demoRun) return;
      loadingStage.textContent = step.label;
      loadingProgress.style.width = `${step.progress}%`;
      loadingTip.textContent = demoTips[index % demoTips.length];
    }, step.delay));
  });

  demoTimers.push(setTimeout(() => {
    if (runId !== demoRun) return;
    loadingProgress.style.width = '100%';
    panelScroll.classList.remove('is-analyzing');
    panelScroll.scrollTo({ top: 0 });
    rerunButton.disabled = false;
    rerunButton.textContent = 'Run demo again';
    triggerTitle.textContent = '86% look authentic';
    triggerSubtitle.textContent = 'Illustrative result · Tap to view';
    triggerScore.classList.remove('loading');
    triggerScore.textContent = '86';
  }, 5400));
}
