import '@testing-library/jest-dom';

// Polyfill window.scrollTo and Element.prototype.scrollIntoView for JSDOM
window.scrollTo = () => {};
Element.prototype.scrollIntoView = () => {};

// Mock navigator.clipboard
Object.assign(navigator, {
  clipboard: {
    writeText: async () => Promise.resolve(),
  },
});
