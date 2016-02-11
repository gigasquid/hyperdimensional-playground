(ns hyperdimensional-playground.context-vectors
  (:require [clojure.core.matrix :as m]
            [clojure.core.matrix.linear :as ml]
            [clojure.string :as string]
            [hyperdimensional-playground.core :refer [rand-hv cosine-sim mean-add inverse xor-mul]]
            [hyperdimensional-playground.fairytale-nouns :refer [fairy-tales-nouns book-list]]))

(m/set-current-implementation :vectorz)
;; size of the hypervectors and freq matrix columns
(def sz 10000)
;; The nouns come from a sampling of Grimm's fairy tale nouns these will
;; make up the rows in the frequency matrix
(def noun-idx (zipmap fairy-tales-nouns (range)))
(def freq-matrix (m/new-sparse-array [(count fairy-tales-nouns) sz]))

(defn update-doc!
  "Given a document - upate the frequency matrix using random indexing"
  [doc]
  (let [known-nouns (clojure.set/intersection fairy-tales-nouns (set doc))]
    ; use positive random indexing
    (doall (repeatedly 10 #(doseq [noun known-nouns]
                       (m/mset! freq-matrix (get noun-idx noun) (rand-int sz) 1))))))

(defn process-book
  "Load a book and break it into sentence like documents and update the frequency matrix"
  [book-str]
  (let [book-text (slurp book-str)
        docs (partition 25 (map string/lower-case
                                (string/split book-text #"\s|\.|\,|\;|\!|\?")))
        doc-count (count docs)]
    (println "processing:" book-str "(with" doc-count "docs)")
    (doall (map-indexed (fn [idx doc]
                          (when (zero? (mod idx 1000)) (println "doc:" idx))
                          (update-doc! doc))
                        docs))
    (println "DONE with " book-str)))

(defn wv [word]
  "Get a hypervector for the word from the frequency matrix"
  (let [i (get noun-idx word)]
    (assert (not (nil? i)) (str word " not found"))
    (m/slice freq-matrix i)))

(defn compare-wvs
  "Compare two words and give the cosine distance info map"
  [word1 word2]
  (let [wv1 (wv word1)
        wv2 (wv word2)]
    (when (not= word1 word2)
      {:word1 word1
       :word2 word2
       :cosine (cosine-sim wv1 wv2)})))

(defn sim-report
  "Give a report for the first 100 words in the known nouns"
  []
  (for [word1 (take 100 fairy-tales-nouns)
        word2 (take 100 fairy-tales-nouns)]
    (when (not= word1 word2)
      {:word1 word1
       :word2 word2
       :cosine (cosine-sim (m/slice freq-matrix (get noun-idx word1))
                           (m/slice freq-matrix (get noun-idx word2)))})))

(defn sim-for-word
  "Give cosine sim reports for a word for a given sim threshold"
  [word threshold]
  (let [results  (mapv #(compare-wvs word %)
                       fairy-tales-nouns)]
    (->> results
        (filter #(and (:cosine %) (< threshold (:cosine %))))
        (sort-by :cosine)
        (reverse))))

(defn sim-for-vec
  "Give cosine sim reports for a given vector and threshold
  "[v threshold]
  (let [results  (mapv #(hash-map :word %
                                 :cosine (cosine-sim v (wv %)))
                       fairy-tales-nouns)]
    (->> results
        (filter #(and (:cosine %) (< threshold (:cosine %))))
        (sort-by :cosine)
        (reverse))))


(defn unlike-for-word
  "find words unlike that one for a threshold"
  [word threshold]
  (let [results  (mapv #(compare-wvs word %)
                       fairy-tales-nouns)]
    (->> results
        (filter #(and (:cosine %) (> threshold (:cosine %))))
        (sort-by :cosine))))

(defn wv-subtract
  "subract with word vectors
  "[word1 word2]
  (mean-add (wv word1) (inverse (wv word2))))

(defn wv-add
  "add with word vectors"
  [word1 word2]
  (mean-add (wv word1) (wv word2)))

(comment

  ;;; the main loop to run the books and calc the freq matrix
  (doseq [book book-list]
    (process-book book))

  (sim-for-word "prince" 0.5)

  (sort-by :cosine[(compare-wvs "king" "queen")
                   (compare-wvs "king" "prince")
                   (compare-wvs "king" "princess")
                   (compare-wvs "king" "guard")
                   (compare-wvs "king" "goat")])
  ;; ({:word1 "king", :word2 "goat", :cosine 0.1509151478896664}
  ;;  {:word1 "king", :word2 "guard", :cosine 0.16098893367403827}
  ;;  {:word1 "king", :word2 "queen", :cosine 0.49470535530616655}
  ;;  {:word1 "king", :word2 "prince", :cosine 0.5832521795716931}
  ;;  {:word1 "king", :word2 "princess", :cosine 0.5836922474743367})



  (cosine-sim (wv "boy") (wv "king")) ;=> 0.42996397142253145
  (cosine-sim (wv "boy") (wv "goat")) ;=> 0.10263384987428588
  ;; boy + gold = king
  (cosine-sim (mean-add (wv "boy") (wv "gold"))
              (wv "king")) ;=> 0.5876251031366048

  (cosine-sim (wv "boy") (wv "jack")) ;=> 0.33102858702785953
  ;; boy + giant = jack
  (cosine-sim (mean-add (wv "giant") (wv "boy"))
              (wv "jack")) ;=>0.4491473187787431


  (cosine-sim (wv "queen") (wv "woman")) ;=> 0.38827204630798223
  ;;; queen= (king-man) + woman
  (cosine-sim (wv "queen")
              (mean-add (wv "woman") (wv-subtract "king" "man"))) ;=>0.5659832204544486

  ;;; frog + princess = prince
  (cosine-sim (wv-add "frog" "princess") (wv "prince")) ;=> 0.5231641991974249

  ;; father = mother -women + man
  (cosine-sim (wv "father")
              (mean-add (wv "man") (wv-subtract "mother" "woman"))) ;=>0.5959841177719538
  (cosine-sim (wv "father") (wv "woman"));=>
                                        ;0.39586439279718383



  ;;; Reasoning with word vectors
  ;; Expressing facts with word vectors

  ;; hansel is the brother of gretel
  ;; B*H + B*G
  (def hansel-brother-of-gretel
    (mean-add
     (xor-mul (wv "brother") (wv "hansel"))
     (xor-mul (wv "brother") (wv "gretel"))))

  (def jack-brother-of-hansel
    (mean-add
     (xor-mul (wv "brother") (wv "jack"))
     (xor-mul (wv "brother") (wv "hansel"))))

  (def facts (mean-add hansel-brother-of-gretel
                       jack-brother-of-hansel))

  ;; is jack the brother of hansel?
  (cosine-sim
   (wv "jack")
   (xor-mul (mean-add (wv "brother") (wv "gretel"))
            facts)) ;=>0.8095270629815969

  ;; is cinderella the brother of gretel ?
  (cosine-sim
   (wv "cinderella")
   (xor-mul (mean-add (wv "brother") (wv "gretel"))
            facts)) ;=>0.1451799916656951

  ;; is jack the brother of gretel ?
  (cosine-sim
   (wv "jack")
   (xor-mul (mean-add (wv "brother") (wv "gretel"))
            facts)) ;=> 0.8095270629815969


  ;; let's add some more facts

  ;; gretel is the sister of hansel
  ;; S*G + S*H
  (def gretel-sister-of-hansel
    (mean-add
     (xor-mul (wv "sister") (wv "gretel"))
     (xor-mul (wv "sister") (wv "hansel"))))

  ;; gretel is the sister of jack
  ; S*G + S*H
  (def gretel-sister-of-jack
    (mean-add
     (xor-mul (wv "sister") (wv "gretel"))
     (xor-mul (wv "sister") (wv "jack"))))

;;; a sibling is a brother and sister

  (def siblings (rand-hv))
  (def siblings-brother-sister
    (mean-add (xor-mul siblings (wv "brother")) (xor-mul siblings (wv "sister"))))

  (def facts (mean-add hansel-brother-of-gretel
                       jack-brother-of-hansel
                       gretel-sister-of-jack
                       gretel-sister-of-hansel
                       siblings-brother-sister))

  ;; are hansel and gretel siblings?
  (cosine-sim
   (mean-add (wv "hansel") (wv "gretel"))
   (xor-mul siblings facts)) ;=>0.6270153790340673

  ;; are john and cinderella siblings?
  (cosine-sim
   (mean-add (wv "roland") (wv "john"))
   (xor-mul siblings facts)) ;=> 0.1984017637065277

  ;; are jack and gretel siblings?
  (cosine-sim
   (mean-add (wv "jack") (wv "gretel"))
   (xor-mul siblings facts)) ;=> 0.47552759147308354

  ;; are jack and hansel siblings?
  (cosine-sim
   (mean-add (wv "jack") (wv "hansel"))
   (xor-mul siblings facts)) ;=>0.48003572523507465


  ;; What about retracting a facts?
  ;; let's retract the fact that jack is the brother of hansel
  ;; and that jack is the sister of gretel
  (def new-facts (mean-add facts
                           (inverse jack-brother-of-hansel)
                           (inverse gretel-sister-of-jack)))


 ;; are hansel and gretel siblings?
  (cosine-sim
   (mean-add (wv "hansel") (wv "gretel"))
   (xor-mul siblings new-facts)) ;=> 0.3190193236252822


  ;; are jack and gretel siblings?
  (cosine-sim
   (mean-add (wv "jack") (wv "gretel"))
   (xor-mul siblings new-facts)) ;=>0.18097717659128212

  ;; are jack and hansel siblings?
  (cosine-sim
   (mean-add (wv "jack") (wv "hansel"))
   (xor-mul siblings new-facts)) ;=>0.17644485503854168

  ;; are jack and cinderella siblings?
  (cosine-sim
   (mean-add (wv "jack") (wv "roland"))
   (xor-mul siblings new-facts)) ;=> 0.11292438924185935

  )
