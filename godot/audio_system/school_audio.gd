extends Node
## SchoolAudio - the adaptive horror-audio director (autoload).
##
## The building is listening. This node owns every sound that is not a character's own Foley, decides what the
## school does next from the current psychological state, remembers what the player has already heard, and
## responds to what the player does (walking, stopping, running, entering rooms, looking away).
##
## Intensity states change the *meaning* of the environment, not just its volume:
##   0 FALSE_SAFETY  the school teaches the player its ordinary sounds (rain, hum, clock, pipes, drips)
##   1 WRONG         those sounds start to misbehave (clock hesitates, the PA opens on a room full of children)
##   2 OBSERVED      sounds answer the player (counting synced to their steps, a copy of their footsteps)
##   3 DESTABILISED  music, voice and building blur (the vent breathes, the bell won't stop, the register)
##   4 PRESENCE      the hum dies, the rain is sealed out, something is at the player's ears
##   5 AFTERMATH     almost nothing. Then something polite. Then the clocks start again.
##
## Integration (see docs/INTEGRATION.md):
##   SchoolAudio.set_player(player_node)                    # a Node3D; its global position/basis are read
##   SchoolAudio.footstep(surface)                          # call from your controller on every step
##   SchoolAudio.set_moving(is_moving, is_running)
##   SchoolAudio.enter_room(room_kind)                      # or use RoomZone (audio_system/room_zone.gd)
##   SchoolAudio.set_intensity(n) / SchoolAudio.push_intensity(n) # scripted narrative control
##   SchoolAudio.register_emitter(kind, node3d)             # or use AudioEmitter markers
##   SchoolAudio.play_sequence("register")                  # scripted set pieces

signal intensity_changed(old_state: int, new_state: int)
signal motif_played(motif: String, detail: Dictionary)

enum State { FALSE_SAFETY, WRONG, OBSERVED, DESTABILISED, PRESENCE, AFTERMATH }

const MANIFEST := "res://audio/manifest.json"
const BUSES := ["Env", "World", "Voice", "Head", "Music"]
const ROOMS := ["classroom", "corridor", "stairwell", "hall", "music_room"]
## Reverb buses approximate the offline rooms (decay is what matters; the offline renders are the reference).
const ROOM_REVERB := {
	"classroom": {"room_size": 0.55, "damping": 0.35, "spread": 0.9, "wet": 0.32, "predelay_msec": 12.0, "hipass": 0.05},
	"music_room": {"room_size": 0.5, "damping": 0.45, "spread": 0.9, "wet": 0.28, "predelay_msec": 10.0, "hipass": 0.05},
	"corridor": {"room_size": 0.78, "damping": 0.25, "spread": 0.7, "wet": 0.42, "predelay_msec": 18.0, "hipass": 0.08},
	"stairwell": {"room_size": 0.88, "damping": 0.2, "spread": 1.0, "wet": 0.5, "predelay_msec": 25.0, "hipass": 0.08},
	"hall": {"room_size": 0.95, "damping": 0.3, "spread": 1.0, "wet": 0.55, "predelay_msec": 40.0, "hipass": 0.05},
}

## Per-state behaviour. Gains in dB relative to calibrated asset level. rates are mean seconds between events.
const PROFILE := {
	State.FALSE_SAFETY: {"rain": 0.0, "roomtone": 0.0, "hum": 0.0, "vent": -6.0, "vent_breathing": -80.0,
		"strings_far": -80.0, "strings_near": -80.0, "bowed": -80.0, "env_cutoff": 20000.0,
		"events": {"drip": 4.0, "pipe_knocks": 90.0, "floor_creak": 25.0, "distant_door": 120.0}},
	State.WRONG: {"rain": -1.0, "roomtone": 0.0, "hum": 0.0, "vent": -6.0, "vent_breathing": -80.0,
		"strings_far": -18.0, "strings_near": -80.0, "bowed": -80.0, "env_cutoff": 20000.0,
		"events": {"drip": 4.0, "pipe_knocks": 70.0, "floor_creak": 25.0, "clock_stutter": 60.0, "pa_carrier": 150.0,
			"piano_exercise": 120.0, "distant_door": 150.0}},
	State.OBSERVED: {"rain": -3.0, "roomtone": -2.0, "hum": 0.0, "vent": -8.0, "vent_breathing": -80.0,
		"strings_far": -10.0, "strings_near": -80.0, "bowed": -20.0, "env_cutoff": 16000.0,
		"events": {"drip": 5.0, "door_knocks": 80.0, "chalk": 120.0, "hum_dropout": 70.0, "piano_exercise": 100.0,
			"bell_far": 240.0, "floor_creak": 30.0, "silence_window": 120.0}},
	State.DESTABILISED: {"rain": -5.0, "roomtone": -4.0, "hum": -3.0, "vent": -20.0, "vent_breathing": -6.0,
		"strings_far": -6.0, "strings_near": -16.0, "bowed": -10.0, "env_cutoff": 9000.0,
		"events": {"drip": 6.0, "chair_ceiling": 70.0, "bell_wrong": 200.0, "hum_dropout": 40.0, "whisper_count": 60.0,
			"door_knocks": 90.0, "silence_window": 90.0}},
	State.PRESENCE: {"rain": -14.0, "roomtone": -12.0, "hum": -80.0, "vent": -80.0, "vent_breathing": -3.0,
		"strings_far": -80.0, "strings_near": -2.0, "bowed": -6.0, "env_cutoff": 450.0,
		"events": {"breath": 7.0, "whisper_circle": 18.0, "chair_ceiling": 40.0}},
	State.AFTERMATH: {"rain": -80.0, "roomtone": -22.0, "hum": -80.0, "vent": -80.0, "vent_breathing": -80.0,
		"strings_far": -80.0, "strings_near": -80.0, "bowed": -80.0, "env_cutoff": 700.0,
		"events": {"drip": 20.0}},
}

var state: int = State.FALSE_SAFETY
var room: String = "classroom"
var player: Node3D
var rng := RandomNumberGenerator.new()

var _assets := {}            # id -> manifest entry
var _streams := {}           # id -> AudioStream (lazy)
var _bags := {}              # tag -> shuffled ids (no-repeat selection)
var _cooldown_until := {}    # id/motif -> seconds
var _beds := {}              # name -> AudioStreamPlayer
var _bed_target_db := {}
var _emitters := {}          # kind -> Array[Node3D]
var _clock_running := true
var _clock_phase := 0.0
var _clock_tick := 0
var _next_event := {}        # event -> seconds
var _time := 0.0
var _silence_until := 0.0
var _moving := false
var _running := false
var _last_step_time := -100.0
var _stopped_since := 0.0
var _pending_extra_step := false
var _env_lowpass: AudioEffectLowPassFilter

## What the player has heard. Motifs read and write this; it is how the building "remembers".
var memory := {
	"knock_patterns": 0,        # how many complete three-knock patterns have played
	"knock_stage": "three",     # three -> two -> behind
	"count": 0,                 # the counting child's last number (persists across rooms)
	"counted_extra": false,     # has the child already taken a step the player didn't?
	"register_done": false,
	"answered_behind": false,
	"hymn_heard": 0,
	"thank_you_done": false,
	"stops": 0,                 # times the player stopped walking
	"runs": 0,
	"rooms_entered": {},
	"anomalies": [],            # [time, name] log, newest last
}


func _ready() -> void:
	rng.seed = 1983
	_setup_buses()
	_load_manifest()
	_start_beds()
	_apply_profile(true)
	process_mode = Node.PROCESS_MODE_ALWAYS


# ------------------------------------------------------------------ public API

func set_player(node: Node3D) -> void:
	player = node


func set_intensity(new_state: int) -> void:
	new_state = clampi(new_state, State.FALSE_SAFETY, State.AFTERMATH)
	if new_state == state:
		return
	var old := state
	state = new_state
	_on_state_entered(old, new_state)
	_apply_profile(false)
	intensity_changed.emit(old, new_state)


## Raise intensity only (narrative beats that should never calm the game down).
func push_intensity(new_state: int) -> void:
	if new_state > state:
		set_intensity(new_state)


func enter_room(kind: String) -> void:
	if not ROOMS.has(kind) or kind == room:
		return
	room = kind
	memory["rooms_entered"][kind] = memory["rooms_entered"].get(kind, 0) + 1
	_set_room_reverb(kind)
	_crossfade_room_beds()


func set_moving(is_moving: bool, is_running: bool = false) -> void:
	if _moving and not is_moving:
		memory["stops"] += 1
		_stopped_since = _time
		_on_player_stopped()
	if is_running and not _running:
		memory["runs"] += 1
		_on_player_ran()
	_moving = is_moving
	_running = is_running


## Call on every footstep of the player. Plays the step and lets the building answer it.
func footstep(surface: String = "lino") -> void:
	var id := _pick("step_" + surface)
	if id != "":
		_play_at_player(id, "World", -2.0, Vector3(0, -1.5, 0))
	_last_step_time = _time
	_answer_footstep()


func register_emitter(kind: String, node: Node3D) -> void:
	if not _emitters.has(kind):
		_emitters[kind] = []
	_emitters[kind].append(node)
	if kind == "hum" and _beds.has("hum"):
		pass  # hum fittings get their own 3D loop
	if kind == "hum":
		_attach_loop(node, "beds.hum_tube", "Env", PROFILE[state]["hum"])


func play_sequence(name: String) -> void:
	match name:
		"register": _seq_register()
		"hymn": _seq_hymn()
		"presence_climax": _seq_climax()
		"thank_you": _seq_thank_you()
		_: push_warning("SchoolAudio: unknown sequence " + name)


# ------------------------------------------------------------------ main loop

func _process(delta: float) -> void:
	_time += delta
	_tick_master_clock(delta)
	_update_bed_gains(delta)
	if _pending_extra_step and _time - _last_step_time > 0.62 and not _moving:
		_pending_extra_step = false
		_child_counts(true)
	if _time < _silence_until:
		return
	var events: Dictionary = PROFILE[state]["events"]
	for ev in events:
		if not _next_event.has(ev):
			_next_event[ev] = _time + _exp(events[ev])
		elif _time >= _next_event[ev]:
			_next_event[ev] = _time + _exp(events[ev])
			_fire(ev)


func _exp(mean: float) -> float:
	# exponential inter-arrival, floored: unpredictable but never clustered into noise
	return max(mean * 0.25, -log(max(rng.randf(), 1e-4)) * mean)


func _fire(ev: String) -> void:
	match ev:
		"drip": _play_world(_pick("drip"), _emitter_pos("drip", 6.0), "World", -4.0)
		"pipe_knocks": _knocks("pipe")
		"door_knocks": _knocks("door")
		"floor_creak": _play_world(_pick("floor_creak"), _random_around(8.0), "World", -10.0)
		"distant_door": _play_world(_pick("door_slam"), _random_around(25.0), "World", -12.0)
		"clock_stutter": _clock_stutter()
		"pa_carrier": _pa(["voice.pa_carrier_hall"], true)
		"piano_exercise": _piano_exercise()
		"chalk": _play_world(_pick("chalk"), _emitter_pos("door", 6.0), "World", 0.0, "chalk")
		"hum_dropout": _hum_dropout(rng.randf_range(0.8, 3.0))
		"bell_far": _play_world(_pick_prefix("world.bell_natural"), _random_around(30.0, 12.0), "World", -6.0, "bell")
		"bell_wrong": _play_world(_pick_prefix("world.bell_wrong"), _random_around(12.0, 10.0), "World", -4.0, "bell")
		"chair_ceiling": _play_world(_pick("ceiling"), _player_pos() + Vector3(rng.randf_range(-2, 2), 3.2, rng.randf_range(-2, 2)), "World", -2.0, "chair")
		"whisper_count": _play_head("voice.count_whisper_%02d" % clampi(memory["count"] + 1, 1, 10), -6.0)
		"breath": _breath()
		"whisper_circle": _play_head(_pick_prefix("head.whisper_circle"), -2.0)
		"silence_window": _hold_breath(rng.randf_range(20.0, 55.0))
	_log(ev)


# ------------------------------------------------------------------ motifs (with memory)

## The three knocks. Taught as innocent pipe noise, repeated, then a door knocks only twice,
## and the third knock arrives later - right behind the player.
func _knocks(material: String) -> void:
	if _on_cooldown("knocks", 45.0):
		return
	var timing := [0.0, 0.34, 0.97]
	var count := 3
	if memory["knock_stage"] == "two" and material == "door":
		count = 2
	elif memory["knock_patterns"] >= 3 and state >= State.OBSERVED and material == "door":
		memory["knock_stage"] = "two"
		count = 2
	var pos := _emitter_pos(material if material == "door" else "pipe", 5.0)
	for i in count:
		var id := _pick("knock_" + material)
		get_tree().create_timer(timing[i] * rng.randf_range(0.98, 1.02)).timeout.connect(
			_play_world.bind(id, pos, "World", (0.0 if i != 1 else -1.5)))
	if count == 3:
		memory["knock_patterns"] += 1
	motif_played.emit("knocks", {"material": material, "count": count})


## A child counts the player's footsteps from somewhere ahead. In OBSERVED+ it may take a step the player didn't.
func _answer_footstep() -> void:
	if state < State.WRONG or state >= State.PRESENCE:
		return
	if state == State.WRONG and rng.randf() > 0.15 and memory["count"] == 0:
		return
	if memory["count"] >= 9:
		return
	_child_counts(false)
	if state >= State.OBSERVED and not memory["counted_extra"] and memory["count"] >= 6:
		_pending_extra_step = true


func _child_counts(extra: bool) -> void:
	memory["count"] += 1
	if extra:
		memory["counted_extra"] = true
	var n: int = clampi(memory["count"], 1, 10)
	var ahead := _player_forward() * (22.0 - n * (1.2 if extra else 0.7))
	var pos := _player_pos() + ahead + Vector3(0, 0.2, 0)
	get_tree().create_timer(0.09).timeout.connect(_play_world.bind("voice.count_%02d" % n, pos, "Voice", -2.0))
	motif_played.emit("count", {"n": n, "extra": extra})


func _on_player_stopped() -> void:
	# copy footsteps take one more step after the player stops (OBSERVED and DESTABILISED only)
	if state in [State.OBSERVED, State.DESTABILISED] and rng.randf() < 0.35 and not _on_cooldown("copy_step", 40.0):
		var pos := _player_pos() - _player_forward() * rng.randf_range(6.0, 10.0)
		get_tree().create_timer(rng.randf_range(0.5, 0.7)).timeout.connect(
			_play_world.bind(_pick("step_lino"), pos, "World", -6.0))
		_log("copy_step")


func _on_player_ran() -> void:
	# the PA notices
	if state >= State.WRONG and state <= State.DESTABILISED and not _on_cooldown("walk_dont_run", 180.0):
		get_tree().create_timer(rng.randf_range(1.5, 3.0)).timeout.connect(_pa.bind(["voice.pa_walk_dont_run"], false))


func _clock_stutter() -> void:
	_clock_phase = 0.81   # next tick comes ~0.19 s early: the clock hesitates
	_log("clock_stutter")


func _tick_master_clock(delta: float) -> void:
	if not _clock_running:
		return
	_clock_phase += delta
	if _clock_phase >= 1.0:
		_clock_phase -= 1.0
		_clock_tick += 1
		for node in _emitters.get("clock", []):
			_play_world("world.clock_tick_%02d" % (_clock_tick % 7), node.global_position, "World", 0.0)


func _hum_dropout(seconds: float) -> void:
	for p in _emitter_players("hum"):
		var t := create_tween()
		t.tween_property(p, "volume_db", -80.0, 0.02)
		t.tween_interval(seconds)
		t.tween_property(p, "volume_db", PROFILE[state]["hum"], 0.02)
	if state >= State.OBSERVED and rng.randf() < 0.3:
		_clock_running = false   # every clock in the school stops with the light
		_log("clocks_stopped")


func _piano_exercise() -> void:
	if _on_cooldown("piano", 90.0):
		return
	var notes := [60, 62, 64, 65]
	if state >= State.OBSERVED and rng.randf() < 0.5:
		notes = [60, 62, 64, 65, 65, 65]   # stuck on F. It never reaches G.
	var pos := _emitter_pos("piano", 20.0)
	var t := 0.05
	for i in notes.size():
		var id := "world.piano_%d_v%d" % [notes[i], rng.randi() % 2]
		get_tree().create_timer(t).timeout.connect(_play_world.bind(id, pos, "Music", -4.0))
		t += 0.66 * rng.randf_range(0.85, 1.25) + (rng.randf_range(0.0, 0.35) if i == notes.size() - 2 else 0.0)
	motif_played.emit("piano_exercise", {"notes": notes})


func _breath() -> void:
	var inhale := _pick_prefix("head.breath_in")
	_play_head(inhale, 0.0)
	get_tree().create_timer(1.6).timeout.connect(_play_head.bind(_pick_prefix("head.breath_out"), -6.0))


## A long stretch where the school deliberately does nothing at all.
func _hold_breath(seconds: float) -> void:
	_silence_until = _time + seconds
	for name in ["rain", "roomtone"]:
		if _bed_target_db.has(name):
			_bed_target_db[name] -= 4.0
	get_tree().create_timer(seconds).timeout.connect(_apply_profile.bind(false))
	_log("silence_window")


# ------------------------------------------------------------------ set pieces

func _seq_register() -> void:
	if memory["register_done"]:
		return
	memory["register_done"] = true
	var lines := [["voice.pa_good_morning", 2.6], ["voice.pa_name_thomas_avery", 1.9], ["voice.pa_answer_00", 1.5],
		["voice.pa_name_hannah_bell", 1.9], ["voice.pa_answer_01", 1.5], ["voice.pa_name_daniel_cole", 1.9],
		["voice.pa_answer_02", 1.5], ["voice.pa_name_sophie_doyle", 4.2], ["voice.pa_name_sophie_doyle_q", 3.4]]
	var t := 0.6
	_pa_click(true)
	for l in lines:
		get_tree().create_timer(t).timeout.connect(_pa.bind([l[0]], false, false))
		t += l[1]
	# the answer is not on the PA. It is in the corridor, behind the player.
	get_tree().create_timer(t).timeout.connect(_answer_behind)
	get_tree().create_timer(t + 2.6).timeout.connect(_pa_click.bind(false))


func _answer_behind() -> void:
	memory["answered_behind"] = true
	_play_world("voice.child_here_01", _player_pos() - _player_forward() * 9.0, "Voice", 0.0)
	motif_played.emit("answered_behind", {})


func _seq_hymn() -> void:
	memory["hymn_heard"] += 1
	var id := "music.hymn" if memory["hymn_heard"] == 1 else "music.hymn_one_voice"
	_play_world("music.assembly_hush", _emitter_pos("hall", 25.0), "Music", 0.0)
	get_tree().create_timer(16.0).timeout.connect(_play_world.bind(id, _emitter_pos("hall", 25.0), "Music", 2.0))


func _seq_climax() -> void:
	# PRESENCE -> the inhale -> cut -> the third knock, behind. Then AFTERMATH.
	set_intensity(State.PRESENCE)
	_pa_click(true, true)
	get_tree().create_timer(0.6).timeout.connect(_pa.bind(["voice.pa_anyone_freezing"], false, false, true))
	get_tree().create_timer(6.0).timeout.connect(_play_head.bind("music.inhale_swell", 0.0))
	get_tree().create_timer(9.2).timeout.connect(_cut_to_silence)
	if memory["knock_stage"] == "two" or memory["knock_patterns"] >= 2:
		get_tree().create_timer(10.85).timeout.connect(_play_head.bind("head.knock_behind", 0.0))
	get_tree().create_timer(30.0).timeout.connect(_seq_thank_you)


func _cut_to_silence() -> void:
	for b in BUSES:
		var idx := AudioServer.get_bus_index(b)
		AudioServer.set_bus_mute(idx, true)
	var hd := AudioServer.get_bus_index("Head")
	AudioServer.set_bus_mute(hd, false)
	for p in get_tree().get_nodes_in_group("school_audio_oneshot"):
		p.stop()
	set_intensity(State.AFTERMATH)
	get_tree().create_timer(1.0).timeout.connect(_unmute_all)


func _unmute_all() -> void:
	for b in BUSES:
		AudioServer.set_bus_mute(AudioServer.get_bus_index(b), false)


func _seq_thank_you() -> void:
	if memory["thank_you_done"]:
		return
	memory["thank_you_done"] = true
	_pa_click(true, true)
	get_tree().create_timer(0.8).timeout.connect(_pa.bind(["voice.pa_thank_you"], false, false, true))
	get_tree().create_timer(2.3).timeout.connect(_pa.bind(["voice.pa_everyone_here"], false, false, true))
	get_tree().create_timer(5.0).timeout.connect(_pa_click.bind(false, true))
	get_tree().create_timer(8.0).timeout.connect(func(): _clock_running = true; _clock_phase = 0.0)


# ------------------------------------------------------------------ state transitions

func _on_state_entered(old: int, new: int) -> void:
	if new == State.PRESENCE:
		_clock_running = false
		_hum_dropout(9999.0)
	if new == State.AFTERMATH:
		_clock_running = false
	if old == State.AFTERMATH and new < State.AFTERMATH:
		_clock_running = true
	_next_event.clear()


func _apply_profile(instant: bool) -> void:
	var p: Dictionary = PROFILE[state]
	for name in ["rain", "roomtone", "vent", "vent_breathing", "strings_far", "strings_near", "bowed"]:
		_bed_target_db[name] = p[name]
	if instant:
		for name in _bed_target_db:
			if _beds.has(name):
				_beds[name].volume_db = _bed_target_db[name]
	for pl in _emitter_players("hum"):
		create_tween().tween_property(pl, "volume_db", p["hum"], 0.03 if p["hum"] < -60 else 2.0)
	if _env_lowpass:
		create_tween().tween_method(func(v): _env_lowpass.cutoff_hz = v, _env_lowpass.cutoff_hz, p["env_cutoff"],
			0.2 if instant else 6.0)


func _update_bed_gains(delta: float) -> void:
	for name in _bed_target_db:
		if _beds.has(name):
			var pl: AudioStreamPlayer = _beds[name]
			pl.volume_db = move_toward(pl.volume_db, _bed_target_db[name], delta * 6.0)


# ------------------------------------------------------------------ setup

func _setup_buses() -> void:
	for b in BUSES:
		if AudioServer.get_bus_index(b) == -1:
			AudioServer.add_bus()
			var idx := AudioServer.bus_count - 1
			AudioServer.set_bus_name(idx, b)
			AudioServer.set_bus_send(idx, "Master")
	_env_lowpass = AudioEffectLowPassFilter.new()
	_env_lowpass.cutoff_hz = 20000.0
	AudioServer.add_bus_effect(AudioServer.get_bus_index("Env"), _env_lowpass)
	# one reverb bus per room; World and Voice players are routed through the current room's reverb
	for r in ROOMS:
		var name := "Rev_" + r
		if AudioServer.get_bus_index(name) == -1:
			AudioServer.add_bus()
			var idx := AudioServer.bus_count - 1
			AudioServer.set_bus_name(idx, name)
			AudioServer.set_bus_send(idx, "Master")
			var rv := AudioEffectReverb.new()
			for k in ROOM_REVERB[r]:
				rv.set(k, ROOM_REVERB[r][k])
			rv.dry = 0.0
			AudioServer.add_bus_effect(idx, rv)
	_set_room_reverb(room)
	var lim := AudioEffectHardLimiter.new()
	lim.ceiling_db = -1.0
	AudioServer.add_bus_effect(AudioServer.get_bus_index("Master"), lim)


func _set_room_reverb(kind: String) -> void:
	# World/Voice send to Master directly and to the room reverb via a parallel bus: emulate by retargeting send.
	for b in ["World", "Voice"]:
		AudioServer.set_bus_send(AudioServer.get_bus_index(b), "Rev_" + kind)
	for r in ROOMS:
		var rv := AudioServer.get_bus_effect(AudioServer.get_bus_index("Rev_" + r), 0) as AudioEffectReverb
		rv.dry = 1.0 if r == kind else 0.0


func _load_manifest() -> void:
	var f := FileAccess.open(MANIFEST, FileAccess.READ)
	if f == null:
		push_error("SchoolAudio: missing " + MANIFEST + " - run pipeline/build.py library")
		return
	var data: Dictionary = JSON.parse_string(f.get_as_text())
	for a in data["assets"]:
		_assets[a["id"]] = a


func get_stream(id: String) -> AudioStream:
	if _streams.has(id):
		return _streams[id]
	if not _assets.has(id):
		push_warning("SchoolAudio: unknown asset " + id)
		return null
	var s: AudioStream = load("res://audio/" + _assets[id]["file"])
	if s is AudioStreamOggVorbis:
		(s as AudioStreamOggVorbis).loop = _assets[id]["loop"]
	elif s is AudioStreamWAV and _assets[id]["loop"]:
		var w := s as AudioStreamWAV
		w.loop_mode = AudioStreamWAV.LOOP_FORWARD
		w.loop_end = int(_assets[id]["seconds"] * 48000.0)
	_streams[id] = s
	return s


func asset_ids() -> Array:
	return _assets.keys()


func _start_beds() -> void:
	var beds := {"rain": "beds.rain_" + _bed_room("rain"), "roomtone": "beds.roomtone_" + _bed_room("roomtone"),
		"vent": "beds.vent", "vent_breathing": "beds.vent_breathing", "strings_far": "music.strings_far",
		"strings_near": "music.strings_near", "bowed": "music.bowed_brake"}
	for name in beds:
		var p := AudioStreamPlayer.new()
		p.stream = get_stream(beds[name])
		p.bus = "Music" if beds[name].begins_with("music") else "Env"
		p.volume_db = -80.0
		add_child(p)
		if p.stream:
			p.play(rng.randf_range(0.0, float(_assets[beds[name]]["seconds"]) * 0.9))
		_beds[name] = p


func _bed_room(layer: String) -> String:
	if layer == "rain":
		return room if room in ["classroom", "corridor", "stairwell"] else "corridor"
	return room if room in ["classroom", "corridor", "stairwell", "hall"] else "classroom"


func _crossfade_room_beds() -> void:
	for layer in ["rain", "roomtone"]:
		var id := "beds.%s_%s" % [layer, _bed_room(layer)]
		var old: AudioStreamPlayer = _beds[layer]
		if old.stream == get_stream(id):
			continue
		var p := AudioStreamPlayer.new()
		p.stream = get_stream(id)
		p.bus = "Env"
		p.volume_db = -60.0
		add_child(p)
		p.play(rng.randf_range(0.0, float(_assets[id]["seconds"]) * 0.9))
		_beds[layer] = p
		var t := create_tween()
		t.tween_property(old, "volume_db", -80.0, 3.0)
		t.tween_callback(old.queue_free)


func _attach_loop(node: Node3D, id: String, bus: String, vol: float) -> void:
	var p := AudioStreamPlayer3D.new()
	p.stream = get_stream(id)
	p.bus = bus
	p.volume_db = vol
	p.unit_size = 3.0
	p.attenuation_filter_cutoff_hz = 6000.0
	p.add_to_group("school_audio_emitter_" + "hum")
	node.add_child(p)
	if p.stream:
		p.play(rng.randf_range(0.0, 50.0))


# ------------------------------------------------------------------ playback helpers

func _pick(tag: String) -> String:
	## no-repeat shuffle bag over every asset carrying the tag
	if not _bags.has(tag) or _bags[tag].is_empty():
		var ids := []
		for id in _assets:
			if tag in _assets[id]["tags"] or id.begins_with("world." + tag):
				if state in _assets[id]["states"]:
					ids.append(id)
		ids.shuffle()
		_bags[tag] = ids
	if _bags[tag].is_empty():
		return ""
	return _bags[tag].pop_back()


func _pick_prefix(prefix: String) -> String:
	var ids := []
	for id in _assets:
		if id.begins_with(prefix):
			ids.append(id)
	return "" if ids.is_empty() else ids[rng.randi() % ids.size()]


func _on_cooldown(key: String, seconds: float) -> bool:
	if _cooldown_until.get(key, -1.0) > _time:
		return true
	_cooldown_until[key] = _time + seconds
	return false


func _play_world(id: String, pos: Vector3, bus: String = "World", vol: float = 0.0, cooldown_key: String = "") -> AudioStreamPlayer3D:
	if id == "" or not _assets.has(id):
		return null
	if cooldown_key != "" and _on_cooldown(cooldown_key, float(_assets[id]["cooldown"])):
		return null
	var p := AudioStreamPlayer3D.new()
	p.stream = get_stream(id)
	p.bus = bus
	p.volume_db = vol
	p.unit_size = 2.0
	p.max_db = 3.0
	p.attenuation_filter_cutoff_hz = 9000.0
	p.attenuation_filter_db = -18.0
	p.add_to_group("school_audio_oneshot")
	add_child(p)
	p.global_position = pos
	p.finished.connect(p.queue_free)
	p.play()
	return p


func _play_at_player(id: String, bus: String, vol: float, offset: Vector3) -> void:
	_play_world(id, _player_pos() + offset, bus, vol)


func _play_head(id: String, vol: float = 0.0) -> void:
	if id == "" or not _assets.has(id):
		return
	var p := AudioStreamPlayer.new()
	p.stream = get_stream(id)
	p.bus = "Head"
	p.volume_db = vol
	add_child(p)
	p.finished.connect(p.queue_free)
	p.play()


func _pa(ids: Array, with_clicks: bool = true, _unused: bool = false, near: bool = false) -> void:
	var speakers: Array = _emitters.get("pa", [])
	if speakers.is_empty():
		return
	if with_clicks:
		_pa_click(true, near)
	var dur := 0.0
	for id in ids:
		for s in _nearest(speakers, 1 if near else 3):
			_play_world(id, s.global_position, "Voice", 0.0)
		dur += float(_assets.get(id, {"seconds": 0.0})["seconds"])
	if with_clicks:
		get_tree().create_timer(dur + 0.6).timeout.connect(_pa_click.bind(false, near))


func _pa_click(on: bool, near: bool = false) -> void:
	var speakers: Array = _emitters.get("pa", [])
	var id := "world.pa_click_%s_%02d" % ["on" if on else "off", rng.randi() % 3]
	for s in _nearest(speakers, 1 if near else 3):
		_play_world(id, s.global_position, "Voice", 0.0)


func _nearest(nodes: Array, count: int) -> Array:
	var sorted := nodes.duplicate()
	var here := _player_pos()
	sorted.sort_custom(func(a, b): return a.global_position.distance_to(here) < b.global_position.distance_to(here))
	return sorted.slice(0, count)


func _emitter_players(kind: String) -> Array:
	return get_tree().get_nodes_in_group("school_audio_emitter_" + kind)


func _emitter_pos(kind: String, fallback_distance: float) -> Vector3:
	var nodes: Array = _emitters.get(kind, [])
	if nodes.is_empty():
		return _random_around(fallback_distance)
	# prefer an emitter the player is NOT looking at
	var best: Node3D = nodes[rng.randi() % nodes.size()]
	for n in nodes:
		if _behind_player(n.global_position):
			best = n
			break
	return best.global_position


func _behind_player(pos: Vector3) -> bool:
	return (pos - _player_pos()).normalized().dot(_player_forward()) < -0.2


func _random_around(distance: float, height: float = 0.0) -> Vector3:
	var a := rng.randf_range(0.0, TAU)
	return _player_pos() + Vector3(cos(a) * distance, height, sin(a) * distance)


func _player_pos() -> Vector3:
	return player.global_position if is_instance_valid(player) else Vector3.ZERO


func _player_forward() -> Vector3:
	return -player.global_transform.basis.z if is_instance_valid(player) else Vector3.FORWARD


func _log(name: String) -> void:
	memory["anomalies"].append([snappedf(_time, 0.01), name])
	if memory["anomalies"].size() > 200:
		memory["anomalies"].pop_front()
