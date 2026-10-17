// Generated held-out evaluation summaries. No live review scores.
globalThis.AuthentiCheckPerformance = {
  "schemaVersion": 1,
  "models": [
    {
      "name": "AuthentiCheck",
      "version": "quality_balanced_v5",
      "dataset": "data/test_reviews.csv",
      "split": "data/splits_dataset.json",
      "evaluatedAt": "2026-10-16",
      "reportSha256": "fe4bf3fcbd4cfa100987d12d92a820cd5e6db4944122ef656a3a527789091791",
      "reviewCount": 230,
      "productCount": 4,
      "testEvaluation": {
        "partition": "test",
        "targetColumn": "ground_truth",
        "comparisonCount": 230,
        "samples": [
          {
            "id": "review_00298",
            "groundTruth": "authentic",
            "prediction": "authentic",
            "correct": true
          },
          {
            "id": "review_00299",
            "groundTruth": "authentic",
            "prediction": "authentic",
            "correct": true
          },
          {
            "id": "review_00300",
            "groundTruth": "authentic",
            "prediction": "authentic",
            "correct": true
          },
          {
            "id": "review_00301",
            "groundTruth": "authentic",
            "prediction": "authentic",
            "correct": true
          },
          {
            "id": "review_00302",
            "groundTruth": "authentic",
            "prediction": "authentic",
            "correct": true
          },
          {
            "id": "review_00303",
            "groundTruth": "authentic",
            "prediction": "authentic",
            "correct": true
          },
          {
            "id": "review_00304",
            "groundTruth": "authentic",
            "prediction": "authentic",
            "correct": true
          },
          {
            "id": "review_00305",
            "groundTruth": "authentic",
            "prediction": "liv",
            "correct": false
          },
          {
            "id": "review_00306",
            "groundTruth": "authentic",
            "prediction": "authentic",
            "correct": true
          },
          {
            "id": "review_00307",
            "groundTruth": "authentic",
            "prediction": "authentic",
            "correct": true
          },
          {
            "id": "review_00308",
            "groundTruth": "authentic",
            "prediction": "authentic",
            "correct": true
          },
          {
            "id": "review_00309",
            "groundTruth": "liv",
            "prediction": "authentic",
            "correct": false
          },
          {
            "id": "review_00310",
            "groundTruth": "authentic",
            "prediction": "liv",
            "correct": false
          },
          {
            "id": "review_00311",
            "groundTruth": "authentic",
            "prediction": "liv",
            "correct": false
          },
          {
            "id": "review_00312",
            "groundTruth": "authentic",
            "prediction": "liv",
            "correct": false
          },
          {
            "id": "review_00313",
            "groundTruth": "authentic",
            "prediction": "liv",
            "correct": false
          },
          {
            "id": "review_00314",
            "groundTruth": "liv",
            "prediction": "authentic",
            "correct": false
          },
          {
            "id": "review_00315",
            "groundTruth": "authentic",
            "prediction": "authentic",
            "correct": true
          },
          {
            "id": "review_00316",
            "groundTruth": "authentic",
            "prediction": "liv",
            "correct": false
          },
          {
            "id": "review_00317",
            "groundTruth": "authentic",
            "prediction": "authentic",
            "correct": true
          }
        ]
      },
      "tasks": [
        {
          "name": "Review quality · RoBERTa + XGBoost",
          "scope": "Four quality classes",
          "targetColumn": "ground_truth",
          "metrics": {
            "macro_f1": 0.33911281367976903,
            "macro_precision": 0.32957945478723405,
            "macro_recall": 0.371765525477707,
            "accuracy": 0.6086956521739131,
            "labels": [
              "Authentic",
              "Deceptive",
              "Low-information",
              "Irrelevant"
            ],
            "confusion_matrix": [
              [
                101,
                2,
                54,
                0
              ],
              [
                2,
                1,
                1,
                0
              ],
              [
                22,
                4,
                38,
                0
              ],
              [
                3,
                1,
                1,
                0
              ]
            ],
            "per_class": [
              {
                "name": "Authentic",
                "precision": 0.7890625,
                "recall": 0.643312101910828,
                "f1": 0.7087719298245613,
                "support": 157
              },
              {
                "name": "Deceptive",
                "precision": 0.125,
                "recall": 0.25,
                "f1": 0.16666666666666666,
                "support": 4
              },
              {
                "name": "Low-information",
                "precision": 0.40425531914893614,
                "recall": 0.59375,
                "f1": 0.48101265822784806,
                "support": 64
              },
              {
                "name": "Irrelevant",
                "precision": 0,
                "recall": 0,
                "f1": 0,
                "support": 5
              }
            ]
          }
        },
        {
          "name": "Review quality · RoBERTa only",
          "scope": "Same test reviews; text-only ablation",
          "targetColumn": "ground_truth",
          "metrics": {
            "macro_f1": 0.3659451009485454,
            "macro_precision": 0.37772397094430993,
            "macro_recall": 0.3612410429936306,
            "accuracy": 0.7652173913043478,
            "labels": [
              "Authentic",
              "Deceptive",
              "Low-information",
              "Irrelevant"
            ],
            "confusion_matrix": [
              [
                141,
                3,
                13,
                0
              ],
              [
                4,
                0,
                0,
                0
              ],
              [
                28,
                1,
                35,
                0
              ],
              [
                4,
                0,
                1,
                0
              ]
            ],
            "per_class": [
              {
                "name": "Authentic",
                "precision": 0.7966101694915254,
                "recall": 0.8980891719745223,
                "f1": 0.8443113772455089,
                "support": 157
              },
              {
                "name": "Deceptive",
                "precision": 0,
                "recall": 0,
                "f1": 0,
                "support": 4
              },
              {
                "name": "Low-information",
                "precision": 0.7142857142857143,
                "recall": 0.546875,
                "f1": 0.6194690265486725,
                "support": 64
              },
              {
                "name": "Irrelevant",
                "precision": 0,
                "recall": 0,
                "f1": 0,
                "support": 5
              }
            ]
          }
        },
        {
          "name": "RoBERTa · text target",
          "scope": "Text classifier evaluated against text_label",
          "targetColumn": "text_label",
          "metrics": {
            "macro_f1": 0.4428289608526337,
            "macro_precision": 0.4620445635881471,
            "macro_recall": 0.43688505203405864,
            "accuracy": 0.7913043478260869,
            "labels": [
              "Authentic",
              "Deceptive",
              "Low-information",
              "Irrelevant"
            ],
            "confusion_matrix": [
              [
                142,
                0,
                9,
                0
              ],
              [
                3,
                1,
                0,
                0
              ],
              [
                28,
                3,
                39,
                0
              ],
              [
                4,
                0,
                1,
                0
              ]
            ],
            "per_class": [
              {
                "name": "Authentic",
                "precision": 0.8022598870056498,
                "recall": 0.9403973509933775,
                "f1": 0.8658536585365854,
                "support": 151
              },
              {
                "name": "Deceptive",
                "precision": 0.25,
                "recall": 0.25,
                "f1": 0.25,
                "support": 4
              },
              {
                "name": "Low-information",
                "precision": 0.7959183673469388,
                "recall": 0.5571428571428572,
                "f1": 0.6554621848739496,
                "support": 70
              },
              {
                "name": "Irrelevant",
                "precision": 0,
                "recall": 0,
                "f1": 0,
                "support": 5
              }
            ]
          }
        },
        {
          "name": "Aspect detection",
          "scope": "Ten aspects · gold Authentic reviews",
          "metrics": {
            "macro_f1": 0.5619272783210519,
            "macro_precision": 0.5573999146928339,
            "macro_recall": 0.5794440898423925,
            "accuracy": 0.08917197452229299
          }
        },
        {
          "name": "Sentiment",
          "scope": "Gold categories · gold Authentic reviews",
          "metrics": {
            "macro_f1": 0.5555314491846736,
            "macro_precision": 0.5458839939912321,
            "macro_recall": 0.6083463913294905,
            "accuracy": 0.5793650793650794
          }
        }
      ]
    }
  ]
};
