import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";

const context = { URL, location: { href: "https://shopee.ph/product/123", hostname: "shopee.ph" } };
vm.createContext(context);
vm.runInContext(readFileSync("extension/extractors/buyer-reviews.js", "utf8"), context);
const { extractUsername } = context.AuthentiCheckExtractors.common;
function candidate(text, hidden = false) {
  return {
    closest: () => hidden ? {} : null,
    cloneNode: () => ({ textContent: text, querySelectorAll: () => [] })
  };
}
const seller = candidate("Seller account");
const buyer = candidate("  m***a  ");
const root = { querySelectorAll: selector => selector === ".author" ? [seller, buyer] : [] };
assert.equal(extractUsername(root, [".author"], [{ contains: element => element === seller }]), "m***a");
assert.equal(extractUsername({ querySelectorAll: () => [candidate("by Juan D.")] }, [".author"]), "Juan D.");
assert.equal(extractUsername({ querySelectorAll: () => [candidate("Hidden buyer", true), candidate("真实买家")] }, [".author"]), "真实买家");
assert.equal(extractUsername({ querySelectorAll: () => [candidate("x".repeat(101))] }, [".author"]), "");
assert.equal(extractUsername({ querySelectorAll: () => [] }, [".author"]), "");
const rating = {};
const plain = Object.assign(candidate("zhaoxi18"), { children: [], compareDocumentPosition: () => 4 });
const masked = Object.assign(candidate("j*****18"), { children: [], compareDocumentPosition: () => 4 });
const config = { ratingSelectors: [".rating"], contentSelectors: [".body"] };
const obfuscatedRoot = {
  querySelector: selector => selector === ".rating" ? rating : null,
  querySelectorAll: selector => selector === "a, span, div" ? [plain] : []
};
assert.equal(extractUsername(obfuscatedRoot, [".known-author"], [], config), "zhaoxi18");
obfuscatedRoot.querySelectorAll = selector => selector === "a, span, div" ? [masked] : [];
assert.equal(extractUsername(obfuscatedRoot, [], [], config), "j*****18");
masked.compareDocumentPosition = () => 2;
assert.equal(extractUsername(obfuscatedRoot, [], [], config), "", "Do not treat arbitrary text after the stars as a name");
const profile = Object.assign(candidate("j*****18"), { getAttribute: () => "/private_handle" });
const profileRoot = { querySelectorAll: selector => selector === "a[href]" ? [profile] : [], querySelector: () => null };
assert.equal(extractUsername(profileRoot, [], [], config), "j*****18", "Use the visible mask, never the profile URL identity");
const body = { contains: element => element === profile };
profileRoot.querySelectorAll = selector => selector === "a[href]" ? [profile] : selector === ".body" ? [body] : [];
assert.equal(extractUsername(profileRoot, [], [], config), "", "Ignore profile links embedded in review prose");
profileRoot.querySelectorAll = selector => selector === "a[href]" ? [profile] : [];
assert.equal(extractUsername(profileRoot, [], [{ contains: element => element === profile }], config), "", "Never use a seller reply's profile");
console.log("Reviewer username extraction preserves masked names and excludes seller/hidden names.");
