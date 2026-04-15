/**
 * Raccourcis clavier pour task-assistant.
 *
 * n       → focus input conversationnel
 * /       → focus input conversationnel (alias)
 * j / k   → naviguer vers la card suivante / précédente
 * x       → toggle done sur la card focalisée
 * g t     → aller à Today  (/)
 * g w     → aller à Week   (/week)
 * g i     → aller à Inbox  (/inbox)
 * ?       → toggle overlay d'aide
 * Esc     → fermer l'overlay d'aide
 */

(function () {
  "use strict";

  // -- helpers ---------------------------------------------------------------

  /** Retourne true si le focus est dans un champ de saisie. */
  function isEditing() {
    const tag = document.activeElement?.tagName;
    return (
      tag === "INPUT" ||
      tag === "TEXTAREA" ||
      tag === "SELECT" ||
      document.activeElement?.isContentEditable
    );
  }

  /** Toutes les task cards visibles, ordonnées dans le DOM. */
  function cards() {
    return Array.from(document.querySelectorAll("[id^='task-']")).filter(
      (el) => el.offsetParent !== null
    );
  }

  // -- navigation j/k --------------------------------------------------------

  let focusedIndex = -1;

  function setFocus(index) {
    const list = cards();
    if (!list.length) return;
    focusedIndex = Math.max(0, Math.min(index, list.length - 1));
    list.forEach((el, i) =>
      el.classList.toggle("ring-2", i === focusedIndex)
    );
    list[focusedIndex]?.scrollIntoView({ block: "nearest" });
  }

  function moveFocus(delta) {
    const list = cards();
    if (!list.length) return;
    if (focusedIndex < 0) {
      setFocus(delta > 0 ? 0 : list.length - 1);
    } else {
      setFocus(focusedIndex + delta);
    }
  }

  // -- toggle done -----------------------------------------------------------

  function toggleDone() {
    const list = cards();
    if (focusedIndex < 0 || !list[focusedIndex]) return;
    const card = list[focusedIndex];
    // Cherche le bouton open→done (x-on:click, pas dans un <form>)
    // ou le bouton done→undo (dans un <form> hx-post).
    const btn =
      card.querySelector("button[title='Marquer comme fait']") ||
      card.querySelector("button[title='Annuler done']");
    if (btn) btn.click();
  }

  // -- g-chord ---------------------------------------------------------------

  let gPending = false;
  let gTimer = null;

  function handleGChord(key) {
    clearTimeout(gTimer);
    gPending = false;
    if (key === "t") {
      window.location.href = "/";
    } else if (key === "w") {
      window.location.href = "/week";
    } else if (key === "i") {
      window.location.href = "/inbox";
    }
  }

  // -- overlay d'aide --------------------------------------------------------

  function toggleHelp() {
    const overlay = document.getElementById("kb-help-overlay");
    if (overlay) {
      overlay.classList.toggle("hidden");
    }
  }

  // -- listener principal ----------------------------------------------------

  document.addEventListener("keydown", function (e) {
    // Ne pas intercepter si modificateurs actifs (sauf Shift pour ?)
    if (e.ctrlKey || e.metaKey || e.altKey) return;

    // Esc : ferme overlay
    if (e.key === "Escape") {
      document.getElementById("kb-help-overlay")?.classList.add("hidden");
      return;
    }

    // ? : toggle aide (même en édition)
    if (e.key === "?" && !isEditing()) {
      e.preventDefault();
      toggleHelp();
      return;
    }

    // Tout le reste : ignorer en mode édition
    if (isEditing()) return;

    // Chord g…
    if (gPending) {
      e.preventDefault();
      handleGChord(e.key);
      return;
    }

    switch (e.key) {
      case "n":
      case "/":
        e.preventDefault();
        document
          .querySelector("textarea[name='message']")
          ?.focus();
        break;

      case "j":
        e.preventDefault();
        moveFocus(1);
        break;

      case "k":
        e.preventDefault();
        moveFocus(-1);
        break;

      case "x":
        e.preventDefault();
        toggleDone();
        break;

      case "g":
        e.preventDefault();
        gPending = true;
        gTimer = setTimeout(() => {
          gPending = false;
        }, 1000);
        break;

      default:
        break;
    }
  });

  // -- réinitialise focusedIndex quand HTMX swape une task card -------------
  // Filtre sur les swaps de cards uniquement pour ne pas perturber la
  // navigation clavier lors de swaps sans rapport (signal, prompt, etc.).
  document.addEventListener("htmx:afterSwap", (e) => {
    const target = e.detail?.target;
    if (target && /^task-\d+$/.test(target.id || "")) {
      cards().forEach((el) => el.classList.remove("ring-2"));
      focusedIndex = -1;
    }
  });
})();
