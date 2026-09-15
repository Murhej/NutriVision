-- NutriVision relational database schema
-- Target: SQLite 3.35+ (also portable to PostgreSQL with minor type edits)
-- Covers:
-- - Authentication/login and account metadata
-- - Profile + profile image
-- - Onboarding answers (goals, challenges, preferences, allergies, activity, exercise, training)
-- - Nutrition planning targets (calories, macros, vitamins/minerals/amino/fatty acids)
-- - Scan and meal logging history
-- - Daily adherence and streak tracking
-- - Food selections chosen by user

-- ------------------------------------------------------------
-- Core account/auth tables
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
  id                  INTEGER PRIMARY KEY,
  public_id           TEXT NOT NULL UNIQUE,          -- UUID/ULID exposed to clients
  username            TEXT NOT NULL UNIQUE,
  email               TEXT NOT NULL UNIQUE,
  password_hash       TEXT NOT NULL,
  is_active           INTEGER NOT NULL DEFAULT 1,
  email_verified      INTEGER NOT NULL DEFAULT 0,
  created_at          TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at          TEXT NOT NULL DEFAULT (datetime('now')),
  last_login_at       TEXT,
  total_scans         INTEGER NOT NULL DEFAULT 0,
  current_streak_days INTEGER NOT NULL DEFAULT 0,
  best_streak_days    INTEGER NOT NULL DEFAULT 0,
  CONSTRAINT chk_users_total_scans CHECK (total_scans >= 0),
  CONSTRAINT chk_users_streak_current CHECK (current_streak_days >= 0),
  CONSTRAINT chk_users_streak_best CHECK (best_streak_days >= 0)
);

CREATE TABLE IF NOT EXISTS auth_sessions (
  id                  INTEGER PRIMARY KEY,
  user_id             INTEGER NOT NULL,
  token_hash          TEXT NOT NULL UNIQUE,
  refresh_token_hash  TEXT,
  user_agent          TEXT,
  ip_address          TEXT,
  issued_at           TEXT NOT NULL DEFAULT (datetime('now')),
  expires_at          TEXT NOT NULL,
  revoked_at          TEXT,
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- ------------------------------------------------------------
-- Profile and onboarding
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS user_profiles (
  user_id                 INTEGER PRIMARY KEY,
  full_name               TEXT NOT NULL,
  age                     INTEGER NOT NULL,
  gender                  TEXT NOT NULL,
  country                 TEXT,
  height_value            REAL NOT NULL,
  height_unit             TEXT NOT NULL,              -- cm|meter|ft_in
  weight_value            REAL NOT NULL,
  weight_unit             TEXT NOT NULL,              -- kg|lb
  target_weight_value     REAL,
  target_weight_unit      TEXT,
  profile_image_url       TEXT,
  profile_image_storage   TEXT,                       -- local|s3|gcs
  work_type               TEXT,                       -- sedentary|physical|hybrid|wfh
  created_at              TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at              TEXT NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
  CONSTRAINT chk_profile_age CHECK (age BETWEEN 10 AND 120),
  CONSTRAINT chk_profile_height CHECK (height_value > 0),
  CONSTRAINT chk_profile_weight CHECK (weight_value > 0)
);

CREATE TABLE IF NOT EXISTS onboarding_responses (
  id                      INTEGER PRIMARY KEY,
  user_id                 INTEGER NOT NULL,
  step_key                TEXT NOT NULL,              -- personal_details|goal|challenge|preferences|allergies|activity|exercise|training_setup
  response_json           TEXT NOT NULL,              -- JSON payload for flexible step data
  is_optional_step        INTEGER NOT NULL DEFAULT 0,
  skipped                 INTEGER NOT NULL DEFAULT 0,
  created_at              TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at              TEXT NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
  CONSTRAINT uq_onboarding_step UNIQUE (user_id, step_key)
);

CREATE TABLE IF NOT EXISTS user_goals (
  id                      INTEGER PRIMARY KEY,
  user_id                 INTEGER NOT NULL,
  goal_name               TEXT NOT NULL,
  is_primary              INTEGER NOT NULL DEFAULT 0,
  created_at              TEXT NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS user_challenges (
  id                      INTEGER PRIMARY KEY,
  user_id                 INTEGER NOT NULL,
  challenge_name          TEXT NOT NULL,
  created_at              TEXT NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS dietary_preferences (
  id                      INTEGER PRIMARY KEY,
  user_id                 INTEGER NOT NULL,
  preference_name         TEXT NOT NULL,
  created_at              TEXT NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS user_allergies (
  id                      INTEGER PRIMARY KEY,
  user_id                 INTEGER NOT NULL,
  allergy_name            TEXT NOT NULL,
  severity                TEXT,                       -- mild|moderate|severe
  created_at              TEXT NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS user_activity (
  user_id                 INTEGER PRIMARY KEY,
  activity_level          TEXT NOT NULL,              -- mostly_sitting|light|moderate|active|manual_labor
  physically_active_job   INTEGER NOT NULL DEFAULT 0,
  created_at              TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at              TEXT NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS user_exercise_habits (
  user_id                 INTEGER PRIMARY KEY,
  exercise_level          TEXT NOT NULL,              -- sedentary|lightly_active|moderately_active|very_active|extra_active
  hours_per_week          REAL NOT NULL,
  created_at              TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at              TEXT NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
  CONSTRAINT chk_exercise_hours CHECK (hours_per_week >= 0)
);

CREATE TABLE IF NOT EXISTS training_setup (
  id                      INTEGER PRIMARY KEY,
  user_id                 INTEGER NOT NULL,
  setup_name              TEXT NOT NULL,              -- home|gym|bodyweight|outdoor|etc
  is_selected             INTEGER NOT NULL DEFAULT 1,
  created_at              TEXT NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- ------------------------------------------------------------
-- Nutrition plan and target details
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS nutrition_plans (
  id                      INTEGER PRIMARY KEY,
  user_id                 INTEGER NOT NULL,
  plan_name               TEXT NOT NULL DEFAULT 'default',
  estimated_from_profile  INTEGER NOT NULL DEFAULT 1,
  target_calories         REAL NOT NULL,
  target_protein_g        REAL NOT NULL,
  target_carbs_g          REAL NOT NULL,
  target_fat_g            REAL NOT NULL,
  target_water_ml         REAL NOT NULL,
  is_active               INTEGER NOT NULL DEFAULT 1,
  created_at              TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at              TEXT NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
  CONSTRAINT chk_plan_calories CHECK (target_calories > 0),
  CONSTRAINT chk_plan_protein CHECK (target_protein_g >= 0),
  CONSTRAINT chk_plan_carbs CHECK (target_carbs_g >= 0),
  CONSTRAINT chk_plan_fat CHECK (target_fat_g >= 0),
  CONSTRAINT chk_plan_water CHECK (target_water_ml >= 0)
);

CREATE TABLE IF NOT EXISTS nutrient_targets (
  id                      INTEGER PRIMARY KEY,
  plan_id                 INTEGER NOT NULL,
  nutrient_group          TEXT NOT NULL,              -- vitamins|mineral_major|mineral_trace|fatty_acid|amino_acid|other
  nutrient_name           TEXT NOT NULL,
  target_value            REAL NOT NULL,
  unit                    TEXT NOT NULL,
  info_description        TEXT,                       -- tooltip/explanation text
  created_at              TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at              TEXT NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY (plan_id) REFERENCES nutrition_plans(id) ON DELETE CASCADE,
  CONSTRAINT uq_nutrient_per_plan UNIQUE (plan_id, nutrient_group, nutrient_name)
);

-- ------------------------------------------------------------
-- Food scan, chosen foods, and intake logging
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS food_catalog (
  id                      INTEGER PRIMARY KEY,
  canonical_label         TEXT NOT NULL UNIQUE,       -- normalized label from model/mapper
  display_name            TEXT NOT NULL,
  source                  TEXT NOT NULL DEFAULT 'system',
  created_at              TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS meal_scans (
  id                      INTEGER PRIMARY KEY,
  user_id                 INTEGER NOT NULL,
  scan_image_url          TEXT,
  predicted_label         TEXT,
  predicted_confidence    REAL,
  chosen_food_id          INTEGER,                    -- final food chosen by user
  chosen_label            TEXT,                       -- denormalized fallback
  meal_type               TEXT,                       -- breakfast|lunch|dinner|snack
  scanned_at              TEXT NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
  FOREIGN KEY (chosen_food_id) REFERENCES food_catalog(id) ON DELETE SET NULL,
  CONSTRAINT chk_scan_confidence CHECK (predicted_confidence IS NULL OR (predicted_confidence >= 0 AND predicted_confidence <= 1))
);

CREATE TABLE IF NOT EXISTS meal_logs (
  id                      INTEGER PRIMARY KEY,
  user_id                 INTEGER NOT NULL,
  scan_id                 INTEGER,
  consumed_at             TEXT NOT NULL,
  meal_name               TEXT NOT NULL,
  serving_description     TEXT,
  calories                REAL NOT NULL DEFAULT 0,
  protein_g               REAL NOT NULL DEFAULT 0,
  carbs_g                 REAL NOT NULL DEFAULT 0,
  fat_g                   REAL NOT NULL DEFAULT 0,
  fiber_g                 REAL NOT NULL DEFAULT 0,
  sodium_mg               REAL NOT NULL DEFAULT 0,
  cholesterol_mg          REAL NOT NULL DEFAULT 0,
  user_selected           INTEGER NOT NULL DEFAULT 0,
  created_at              TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at              TEXT NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
  FOREIGN KEY (scan_id) REFERENCES meal_scans(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS selected_foods (
  id                      INTEGER PRIMARY KEY,
  user_id                 INTEGER NOT NULL,
  meal_log_id             INTEGER NOT NULL,
  food_id                 INTEGER,
  selected_label          TEXT NOT NULL,
  selected_at             TEXT NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
  FOREIGN KEY (meal_log_id) REFERENCES meal_logs(id) ON DELETE CASCADE,
  FOREIGN KEY (food_id) REFERENCES food_catalog(id) ON DELETE SET NULL
);

-- ------------------------------------------------------------
-- Daily adherence and streak metrics
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS daily_intake_summary (
  id                      INTEGER PRIMARY KEY,
  user_id                 INTEGER NOT NULL,
  intake_date             TEXT NOT NULL,              -- YYYY-MM-DD
  calories_intake         REAL NOT NULL DEFAULT 0,
  protein_intake_g        REAL NOT NULL DEFAULT 0,
  carbs_intake_g          REAL NOT NULL DEFAULT 0,
  fat_intake_g            REAL NOT NULL DEFAULT 0,
  water_intake_ml         REAL NOT NULL DEFAULT 0,
  calories_target         REAL NOT NULL DEFAULT 0,
  protein_target_g        REAL NOT NULL DEFAULT 0,
  carbs_target_g          REAL NOT NULL DEFAULT 0,
  fat_target_g            REAL NOT NULL DEFAULT 0,
  hit_calories_target     INTEGER NOT NULL DEFAULT 0,
  hit_protein_target      INTEGER NOT NULL DEFAULT 0,
  hit_carbs_target        INTEGER NOT NULL DEFAULT 0,
  hit_fat_target          INTEGER NOT NULL DEFAULT 0,
  all_macro_targets_hit   INTEGER NOT NULL DEFAULT 0,
  created_at              TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at              TEXT NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
  CONSTRAINT uq_daily_intake UNIQUE (user_id, intake_date)
);

CREATE TABLE IF NOT EXISTS streak_history (
  id                      INTEGER PRIMARY KEY,
  user_id                 INTEGER NOT NULL,
  intake_date             TEXT NOT NULL,
  streak_day_number       INTEGER NOT NULL,
  reason                  TEXT,                       -- e.g., hit calories+macros
  created_at              TEXT NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
  CONSTRAINT uq_streak_day UNIQUE (user_id, intake_date),
  CONSTRAINT chk_streak_day_number CHECK (streak_day_number >= 0)
);

-- ------------------------------------------------------------
-- Indexes for performance
-- ------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_auth_sessions_user_id ON auth_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_auth_sessions_expires_at ON auth_sessions(expires_at);

CREATE INDEX IF NOT EXISTS idx_user_goals_user_id ON user_goals(user_id);
CREATE INDEX IF NOT EXISTS idx_user_challenges_user_id ON user_challenges(user_id);
CREATE INDEX IF NOT EXISTS idx_dietary_preferences_user_id ON dietary_preferences(user_id);
CREATE INDEX IF NOT EXISTS idx_user_allergies_user_id ON user_allergies(user_id);
CREATE INDEX IF NOT EXISTS idx_training_setup_user_id ON training_setup(user_id);

CREATE INDEX IF NOT EXISTS idx_nutrition_plans_user_id ON nutrition_plans(user_id);
CREATE INDEX IF NOT EXISTS idx_nutrient_targets_plan_id ON nutrient_targets(plan_id);

CREATE INDEX IF NOT EXISTS idx_meal_scans_user_id ON meal_scans(user_id);
CREATE INDEX IF NOT EXISTS idx_meal_scans_scanned_at ON meal_scans(scanned_at);

CREATE INDEX IF NOT EXISTS idx_meal_logs_user_id ON meal_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_meal_logs_consumed_at ON meal_logs(consumed_at);

CREATE INDEX IF NOT EXISTS idx_selected_foods_user_id ON selected_foods(user_id);
CREATE INDEX IF NOT EXISTS idx_selected_foods_meal_log_id ON selected_foods(meal_log_id);

CREATE INDEX IF NOT EXISTS idx_daily_intake_user_date ON daily_intake_summary(user_id, intake_date);
CREATE INDEX IF NOT EXISTS idx_streak_history_user_date ON streak_history(user_id, intake_date);

